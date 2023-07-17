#/bin/bash

sd=`dirname $0`
cd $sd/../

SRC=`pwd`
OUT="$1"
MNT="$2"

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z $GG_BUILD_LOW_PERF ]; then
        (time ctest --output-on-failure ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# open_llama_3b_v2

function gg_run_open_llama_3b_v2 {
    cd ${SRC}

    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/config.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/resolve/main/tokenizer.model
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/tokenizer_config.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/special_tokens_map.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/resolve/main/pytorch_model.bin
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/generation_config.json

    cd build-ci-release

    set -e

    path_models="../models-mnt/open-llama/3B-v2"

    model_f16="${path_models}/ggml-model-f16.bin"
    model_q4_0="${path_models}/ggml-model-q4_0.bin"
    model_q4_1="${path_models}/ggml-model-q4_1.bin"
    model_q5_0="${path_models}/ggml-model-q5_0.bin"
    model_q5_1="${path_models}/ggml-model-q5_1.bin"

    python3 ../convert.py ${path_models}

    ./bin/quantize ${model_f16} ${model_q4_0} q4_0
    ./bin/quantize ${model_f16} ${model_q4_1} q4_1
    ./bin/quantize ${model_f16} ${model_q5_0} q5_0
    ./bin/quantize ${model_f16} ${model_q5_1} q5_1

    (time ./bin/main --model ${model_f16}  -s 1234 -n 64 -t 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/main --model ${model_q4_0} -s 1234 -n 64 -t 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/main --model ${model_q4_1} -s 1234 -n 64 -t 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/main --model ${model_q5_0} -s 1234 -n 64 -t 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/main --model ${model_q5_1} -s 1234 -n 64 -t 8 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log

    set +e
}

function gg_sum_open_llama_3b_v2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'OpenLLaMA 3B-v2: text generation\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
}

## main

if [ -z $GG_BUILD_LOW_PERF ]; then
    rm -rf ${SRC}/models-mnt

    mkdir -p $(realpath ${MNT}/models)
    ln -sfn ${MNT}/models ${SRC}/models-mnt

    python3 -m pip install -r ${SRC}/requirements.txt
fi

ret=0

test $ret -eq 0 && gg_run ctest_debug
test $ret -eq 0 && gg_run ctest_release

if [ -z $GG_BUILD_LOW_PERF ]; then
    test $ret -eq 0 && gg_run open_llama_3b_v2
fi

exit $ret
