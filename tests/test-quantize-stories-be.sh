#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    printf "Usage: %s <llama-quantize> <target-folder> [--print]\n" "$0"
    exit 1
fi

quantize=$1
folder=$2
print_only=${3:-}

if [ "$(uname -m)" != s390x ]; then
    echo "error: this test needs a big-endian s390x host" >&2
    exit 1
fi

MODEL_URL=https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M.gguf
MODEL_SHA256=61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a

# sha256 of each big-endian output, taken from files that ran coherently on z17
# F32 is the converted input, all other types come from llama-quantize --pure
declare -a outputs=(
    "F32       "
    "F16       "
    "BF16      "
    "Q4_0      "
    "Q4_1      "
    "Q5_0      "
    "Q5_1      "
    "Q8_0      "
    "Q2_K      "
    "Q3_K      "
    "Q4_K      "
    "Q5_K      "
    "Q6_K      "
    "IQ4_NL    "
    "IQ4_XS    "
    "MXFP4_MOE "
)

check_sha256() {
    local name=$1
    local file=$2
    local expected=$3

    local actual
    actual=$(sha256sum "$file" | cut -d ' ' -f 1)
    echo "$name $actual"

    if [ -z "$print_only" ] && [ "$actual" != "$expected" ]; then
        echo "error: sha256 mismatch for $name: expected '$expected', got '$actual'" >&2
        exit 1
    fi
}

mkdir -p "$folder"

model=$folder/stories15M.gguf
if [ ! -f "$model" ]; then
    wget -q -O "$model" "$MODEL_URL"
fi
check_sha256 stories15M.gguf "$model" "$MODEL_SHA256"

f32=$folder/stories15M-be-F32.gguf
cp "$model" "$f32"
echo YES | python3 "$(dirname "$0")/../gguf-py/gguf/scripts/gguf_convert_endian.py" "$f32" big

for entry in "${outputs[@]}"; do
    read -r type expected <<< "$entry"
    out=$folder/stories15M-be-$type.gguf

    if [ "$type" != F32 ]; then
        "$quantize" --pure "$f32" "$out" "$type"
    fi

    check_sha256 "$type" "$out" "$expected"
done
