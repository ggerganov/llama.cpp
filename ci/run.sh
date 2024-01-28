#/bin/bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -f "$OUT/*.log"
rm -f "$OUT/*.exit"
rm -f "$OUT/*.md"

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

CMAKE_EXTRA=""

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DLLAMA_METAL_SHADER_DEBUG=ON"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DLLAMA_CUBLAS=1"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:\n source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi

    CMAKE_EXTRA="${CMAKE_EXTRA} -DLLAMA_SYCL=1 DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL_F16=ON"
fi
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

    (time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                          ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -L main -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

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

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                            ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z ${GG_BUILD_LOW_PERF} ]; then
        (time ctest --output-on-failure -L main ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -L main -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
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

function gg_get_model {
    local gguf_3b="$MNT/models/open-llama/3B-v2/ggml-model-f16.gguf"
    local gguf_7b="$MNT/models/open-llama/7B-v2/ggml-model-f16.gguf"
    if [[ -s $gguf_3b ]]; then
        echo -n "$gguf_3b"
    elif [[ -s $gguf_7b ]]; then
        echo -n "$gguf_7b"
    else
        echo >&2 "No model found. Can't run gg_run_ctest_with_model."
        exit 1
    fi
}

function gg_run_ctest_with_model_debug {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-debug
    set -e
    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log
    set +e
    cd ..
}

function gg_run_ctest_with_model_release {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-release
    set -e
    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log
    set +e
    cd ..
}

function gg_sum_ctest_with_model_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

function gg_sum_ctest_with_model_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in release mode\n'
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

    gg_wget models-mnt/wikitext/ https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip -o models-mnt/wikitext/wikitext-2-raw-v1.zip -d models-mnt/wikitext/
    head -n 60 models-mnt/wikitext/wikitext-2-raw/wiki.test.raw > models-mnt/wikitext/wikitext-2-raw/wiki.test-60.raw

    path_models="../models-mnt/open-llama/3B-v2"
    path_wiki="../models-mnt/wikitext/wikitext-2-raw"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} -DLLAMA_QKK_64=1 .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                                             ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert.py ${path_models}

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"
    model_q4_0="${path_models}/ggml-model-q4_0.gguf"
    model_q4_1="${path_models}/ggml-model-q4_1.gguf"
    model_q5_0="${path_models}/ggml-model-q5_0.gguf"
    model_q5_1="${path_models}/ggml-model-q5_1.gguf"
    model_q2_k="${path_models}/ggml-model-q2_k.gguf"
    model_q3_k="${path_models}/ggml-model-q3_k.gguf"
    model_q4_k="${path_models}/ggml-model-q4_k.gguf"
    model_q5_k="${path_models}/ggml-model-q5_k.gguf"
    model_q6_k="${path_models}/ggml-model-q6_k.gguf"

    wiki_test_60="${path_wiki}/wiki.test-60.raw"

    ./bin/quantize ${model_f16} ${model_q8_0} q8_0
    ./bin/quantize ${model_f16} ${model_q4_0} q4_0
    ./bin/quantize ${model_f16} ${model_q4_1} q4_1
    ./bin/quantize ${model_f16} ${model_q5_0} q5_0
    ./bin/quantize ${model_f16} ${model_q5_1} q5_1
    ./bin/quantize ${model_f16} ${model_q2_k} q2_k
    ./bin/quantize ${model_f16} ${model_q3_k} q3_k
    ./bin/quantize ${model_f16} ${model_q4_k} q4_k
    ./bin/quantize ${model_f16} ${model_q5_k} q5_k
    ./bin/quantize ${model_f16} ${model_q6_k} q6_k

    (time ./bin/main --model ${model_f16}  -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/main --model ${model_q8_0} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/main --model ${model_q4_0} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/main --model ${model_q4_1} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/main --model ${model_q5_0} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/main --model ${model_q5_1} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/main --model ${model_q2_k} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/main --model ${model_q3_k} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/main --model ${model_q4_k} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/main --model ${model_q5_k} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/main --model ${model_q6_k} -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/perplexity --model ${model_f16}  -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/perplexity --model ${model_q8_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/perplexity --model ${model_q4_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/perplexity --model ${model_q4_1} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/perplexity --model ${model_q5_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/perplexity --model ${model_q5_1} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/perplexity --model ${model_q2_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/perplexity --model ${model_q3_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/perplexity --model ${model_q4_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/perplexity --model ${model_q5_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/perplexity --model ${model_q6_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/imatrix --model ${model_f16} -f ${wiki_test_60} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-imatrix.log

    (time ./bin/save-load-state --model ${model_q4_0} ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log

    function check_ppl {
        qnt="$1"
        ppl=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl > 20.0" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: ppl > 20.0)\n' "$qnt" "$ppl"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$ppl"
        return 0
    }

    check_ppl "f16"  "$(cat $OUT/${ci}-tg-f16.log  | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q8_0" "$(cat $OUT/${ci}-tg-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_0" "$(cat $OUT/${ci}-tg-q4_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_1" "$(cat $OUT/${ci}-tg-q4_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_0" "$(cat $OUT/${ci}-tg-q5_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_1" "$(cat $OUT/${ci}-tg-q5_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q2_k" "$(cat $OUT/${ci}-tg-q2_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q3_k" "$(cat $OUT/${ci}-tg-q3_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_k" "$(cat $OUT/${ci}-tg-q4_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_k" "$(cat $OUT/${ci}-tg-q5_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q6_k" "$(cat $OUT/${ci}-tg-q6_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log

    cat $OUT/${ci}-imatrix.log | grep "Final" >> $OUT/${ci}-imatrix-sum.log

    # lora
    function compare_ppl {
        qnt="$1"
        ppl1=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)
        ppl2=$(echo "$3" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl1 < $ppl2" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: %s > %s)\n' "$qnt" "$ppl" "$ppl1" "$ppl2"
            return 20
        fi

        printf '  - %s @ %s %s OK\n' "$qnt" "$ppl1" "$ppl2"
        return 0
    }

    path_lora="../models-mnt/open-llama/3B-v2/lora"
    path_shakespeare="../models-mnt/shakespeare"

    shakespeare="${path_shakespeare}/shakespeare.txt"
    lora_shakespeare="${path_lora}/ggml-adapter-model.bin"

    gg_wget ${path_lora} https://huggingface.co/slaren/open_llama_3b_v2_shakespeare_lora/resolve/main/adapter_config.json
    gg_wget ${path_lora} https://huggingface.co/slaren/open_llama_3b_v2_shakespeare_lora/resolve/main/adapter_model.bin
    gg_wget ${path_shakespeare} https://huggingface.co/slaren/open_llama_3b_v2_shakespeare_lora/resolve/main/shakespeare.txt

    python3 ../convert-lora-to-ggml.py ${path_lora}

    # f16
    (time ./bin/perplexity --model ${model_f16} -f ${shakespeare}                            -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-f16.log
    (time ./bin/perplexity --model ${model_f16} -f ${shakespeare} --lora ${lora_shakespeare} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-f16.log
    compare_ppl "f16 shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-f16.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-f16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    # q8_0
    (time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare}                            -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-q8_0.log
    (time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare} --lora ${lora_shakespeare} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-q8_0.log
    compare_ppl "q8_0 shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    # q8_0 + f16 lora-base
    (time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare} --lora ${lora_shakespeare} --lora-base ${model_f16} -c 128 -b 128 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log
    compare_ppl "q8_0 / f16 base shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    set +e
}

function gg_sum_open_llama_3b_v2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'OpenLLaMA 3B-v2:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- perplexity:\n%s\n' "$(cat $OUT/${ci}-ppl.log)"
    gg_printf '- imatrix:\n```\n%s\n```\n' "$(cat $OUT/${ci}-imatrix-sum.log)"
    gg_printf '- lora:\n%s\n' "$(cat $OUT/${ci}-lora-ppl.log)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
    gg_printf '- q2_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q2_k.log)"
    gg_printf '- q3_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q3_k.log)"
    gg_printf '- q4_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_k.log)"
    gg_printf '- q5_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_k.log)"
    gg_printf '- q6_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q6_k.log)"
    gg_printf '- save-load-state: \n```\n%s\n```\n' "$(cat $OUT/${ci}-save-load-state.log)"
    gg_printf '- shakespeare (f16):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-f16.log)"
    gg_printf '- shakespeare (f16 lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-f16.log)"
    gg_printf '- shakespeare (q8_0):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log)"
    gg_printf '- shakespeare (q8_0 lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0.log)"
    gg_printf '- shakespeare (q8_0 / f16 base lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log)"
}

# open_llama_7b_v2
# requires: GG_BUILD_CUDA

function gg_run_open_llama_7b_v2 {
    cd ${SRC}

    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/raw/main/config.json
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/resolve/main/tokenizer.model
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/raw/main/tokenizer_config.json
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/raw/main/special_tokens_map.json
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/raw/main/pytorch_model.bin.index.json
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/resolve/main/pytorch_model-00001-of-00002.bin
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/resolve/main/pytorch_model-00002-of-00002.bin
    gg_wget models-mnt/open-llama/7B-v2/ https://huggingface.co/openlm-research/open_llama_7b_v2/raw/main/generation_config.json

    gg_wget models-mnt/wikitext/ https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip -o models-mnt/wikitext/wikitext-2-raw-v1.zip -d models-mnt/wikitext/

    path_models="../models-mnt/open-llama/7B-v2"
    path_wiki="../models-mnt/wikitext/wikitext-2-raw"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} -DLLAMA_CUBLAS=1 .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                                             ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert.py ${path_models}

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"
    model_q4_0="${path_models}/ggml-model-q4_0.gguf"
    model_q4_1="${path_models}/ggml-model-q4_1.gguf"
    model_q5_0="${path_models}/ggml-model-q5_0.gguf"
    model_q5_1="${path_models}/ggml-model-q5_1.gguf"
    model_q2_k="${path_models}/ggml-model-q2_k.gguf"
    model_q3_k="${path_models}/ggml-model-q3_k.gguf"
    model_q4_k="${path_models}/ggml-model-q4_k.gguf"
    model_q5_k="${path_models}/ggml-model-q5_k.gguf"
    model_q6_k="${path_models}/ggml-model-q6_k.gguf"

    wiki_test="${path_wiki}/wiki.test.raw"

    ./bin/quantize ${model_f16} ${model_q8_0} q8_0
    ./bin/quantize ${model_f16} ${model_q4_0} q4_0
    ./bin/quantize ${model_f16} ${model_q4_1} q4_1
    ./bin/quantize ${model_f16} ${model_q5_0} q5_0
    ./bin/quantize ${model_f16} ${model_q5_1} q5_1
    ./bin/quantize ${model_f16} ${model_q2_k} q2_k
    ./bin/quantize ${model_f16} ${model_q3_k} q3_k
    ./bin/quantize ${model_f16} ${model_q4_k} q4_k
    ./bin/quantize ${model_f16} ${model_q5_k} q5_k
    ./bin/quantize ${model_f16} ${model_q6_k} q6_k

    (time ./bin/main --model ${model_f16}  -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/main --model ${model_q8_0} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/main --model ${model_q4_0} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/main --model ${model_q4_1} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/main --model ${model_q5_0} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/main --model ${model_q5_1} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/main --model ${model_q2_k} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/main --model ${model_q3_k} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/main --model ${model_q4_k} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/main --model ${model_q5_k} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/main --model ${model_q6_k} -t 1 -ngl 999 -s 1234 -n 256 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/perplexity --model ${model_f16}  -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/perplexity --model ${model_q8_0} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/perplexity --model ${model_q4_0} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/perplexity --model ${model_q4_1} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/perplexity --model ${model_q5_0} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/perplexity --model ${model_q5_1} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/perplexity --model ${model_q2_k} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/perplexity --model ${model_q3_k} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/perplexity --model ${model_q4_k} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/perplexity --model ${model_q5_k} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/perplexity --model ${model_q6_k} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/imatrix --model ${model_f16} -f ${wiki_test} -t 1 -ngl 999 -c 2048 -b 512 --chunks 4 ) 2>&1 | tee -a $OUT/${ci}-imatrix.log

    (time ./bin/save-load-state --model ${model_q4_0} ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log

    function check_ppl {
        qnt="$1"
        ppl=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl > 20.0" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: ppl > 20.0)\n' "$qnt" "$ppl"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$ppl"
        return 0
    }

    check_ppl "f16"  "$(cat $OUT/${ci}-tg-f16.log  | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q8_0" "$(cat $OUT/${ci}-tg-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_0" "$(cat $OUT/${ci}-tg-q4_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_1" "$(cat $OUT/${ci}-tg-q4_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_0" "$(cat $OUT/${ci}-tg-q5_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_1" "$(cat $OUT/${ci}-tg-q5_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q2_k" "$(cat $OUT/${ci}-tg-q2_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q3_k" "$(cat $OUT/${ci}-tg-q3_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_k" "$(cat $OUT/${ci}-tg-q4_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_k" "$(cat $OUT/${ci}-tg-q5_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q6_k" "$(cat $OUT/${ci}-tg-q6_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log

    cat $OUT/${ci}-imatrix.log | grep "Final" >> $OUT/${ci}-imatrix-sum.log

    # lora
    function compare_ppl {
        qnt="$1"
        ppl1=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)
        ppl2=$(echo "$3" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl1 < $ppl2" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: %s > %s)\n' "$qnt" "$ppl" "$ppl1" "$ppl2"
            return 20
        fi

        printf '  - %s @ %s %s OK\n' "$qnt" "$ppl1" "$ppl2"
        return 0
    }

    path_lora="../models-mnt/open-llama/7B-v2/lora"
    path_shakespeare="../models-mnt/shakespeare"

    shakespeare="${path_shakespeare}/shakespeare.txt"
    lora_shakespeare="${path_lora}/ggml-adapter-model.bin"

    gg_wget ${path_lora} https://huggingface.co/slaren/open_llama_7b_v2_shakespeare_lora/resolve/main/adapter_config.json
    gg_wget ${path_lora} https://huggingface.co/slaren/open_llama_7b_v2_shakespeare_lora/resolve/main/adapter_model.bin
    gg_wget ${path_shakespeare} https://huggingface.co/slaren/open_llama_7b_v2_shakespeare_lora/resolve/main/shakespeare.txt

    python3 ../convert-lora-to-ggml.py ${path_lora}

    # f16
    (time ./bin/perplexity --model ${model_f16} -f ${shakespeare}                            -t 1 -ngl 999 -c 2048 -b 512 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-f16.log
    (time ./bin/perplexity --model ${model_f16} -f ${shakespeare} --lora ${lora_shakespeare} -t 1 -ngl 999 -c 2048 -b 512 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-f16.log
    compare_ppl "f16 shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-f16.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-f16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    # currently not supported by the CUDA backend
    # q8_0
    #(time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare}                            -t 1 -ngl 999 -c 2048 -b 512 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-q8_0.log
    #(time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare} --lora ${lora_shakespeare} -t 1 -ngl 999 -c 2048 -b 512 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-q8_0.log
    #compare_ppl "q8_0 shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    # q8_0 + f16 lora-base
    #(time ./bin/perplexity --model ${model_q8_0} -f ${shakespeare} --lora ${lora_shakespeare} --lora-base ${model_f16} -t 1 -ngl 999 -c 2048 -b 512 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log
    #compare_ppl "q8_0 / f16 shakespeare" "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log | grep "^\[1\]")" "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-lora-ppl.log

    set +e
}

function gg_sum_open_llama_7b_v2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'OpenLLaMA 7B-v2:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- perplexity:\n%s\n' "$(cat $OUT/${ci}-ppl.log)"
    gg_printf '- imatrix:\n```\n%s\n```\n' "$(cat $OUT/${ci}-imatrix-sum.log)"
    gg_printf '- lora:\n%s\n' "$(cat $OUT/${ci}-lora-ppl.log)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
    gg_printf '- q2_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q2_k.log)"
    gg_printf '- q3_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q3_k.log)"
    gg_printf '- q4_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_k.log)"
    gg_printf '- q5_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_k.log)"
    gg_printf '- q6_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q6_k.log)"
    gg_printf '- save-load-state: \n```\n%s\n```\n' "$(cat $OUT/${ci}-save-load-state.log)"
    gg_printf '- shakespeare (f16):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-f16.log)"
    gg_printf '- shakespeare (f16 lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-f16.log)"
    #gg_printf '- shakespeare (q8_0):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-q8_0.log)"
    #gg_printf '- shakespeare (q8_0 lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0.log)"
    #gg_printf '- shakespeare (q8_0 / f16 base lora):\n```\n%s\n```\n' "$(cat $OUT/${ci}-ppl-shakespeare-lora-q8_0-f16.log)"
}

## main

if [ -z ${GG_BUILD_LOW_PERF} ]; then
    # Create symlink: ./llama.cpp/models-mnt -> $MNT/models/models-mnt
    rm -rf ${SRC}/models-mnt
    mnt_models=${MNT}/models
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt

    # Create a fresh python3 venv and enter it
    python3 -m venv "$MNT/venv"
    source "$MNT/venv/bin/activate"

    pip install -r ${SRC}/requirements.txt --disable-pip-version-check
    pip install --editable gguf-py --disable-pip-version-check
fi

ret=0

test $ret -eq 0 && gg_run ctest_debug
test $ret -eq 0 && gg_run ctest_release

if [ -z ${GG_BUILD_LOW_PERF} ]; then
    if [ -z ${GG_BUILD_VRAM_GB} ] || [ ${GG_BUILD_VRAM_GB} -ge 8 ]; then
        if [ -z ${GG_BUILD_CUDA} ]; then
            test $ret -eq 0 && gg_run open_llama_3b_v2
        else
            test $ret -eq 0 && gg_run open_llama_7b_v2
        fi
        test $ret -eq 0 && gg_run ctest_with_model_debug
        test $ret -eq 0 && gg_run ctest_with_model_release
    fi
fi

exit $ret
