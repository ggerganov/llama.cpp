#!/usr/bin/env bash
# Source-semantics test only; never native hardware acceptance or throughput.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
output=${1:?Usage: test-knc-host.sh fresh-output-directory [3120A|5110P|7120P]}
profile=${2:-7120P}
case $profile in
    3120A) l1=4; l2=16;;
    5110P|7120P) l1=8; l2=32;;
    *) echo 'Unknown KNC profile' >&2; exit 2;;
esac
mkdir -- "$output"
output=$(cd -- "$output" && pwd)
knc=$root/ggml/src/ggml-cpu/arch/knc
python3 "$knc/prefetch.py" --source "$knc" --output "$output" --l1 "$l1" --l2 "$l2" --step 64 --q8 ON
python3 "$root/tests/knc/translate_imci.py" --profile "$profile" "$output/quant-native.S" "$output/ext.c"
python3 "$root/tests/knc/translate_imci.py" --profile "$profile" "$output/ggml-knc-native.S" "$output/q5.c"
${CC:-cc} -std=c11 -O1 -g -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer -no-pie \
    -DGGML_KNC_TEST_EMULATION -DGGML_KNC_TEST_PREFETCH_TRACE -DGGML_KNC_MQ_TEST_STAGING \
    -I"$knc" -I"$root/tests/knc" "$knc/knc-mq.c" "$root/tests/knc/test-staging.c" \
    "$output/ext.c" "$output/q5.c" -lm -o "$output/test-staging"
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "$output/test-staging"
