#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    printf "Usage: %s <llama-quantize> <target-folder> [--no-check]\n" "$0"
    printf "  --no-check  print the sha256 of each output without checking it\n"
    printf "Example: %s ./build/bin/llama-quantize ./build/quantize-test\n" "$0"
    exit 1
fi

quantize=$1
folder=$2
no_check=${3:-}

if [ "$(uname -m)" != s390x ]; then
    echo "error: this test needs a big-endian s390x host" >&2
    exit 1
fi

MODEL_URL=https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M.gguf
MODEL_SHA256=61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a

# sha256 of each big-endian output, taken from files that ran coherently
# F32 is the converted input, all other types come from llama-quantize --pure
declare -a outputs=(
    "F32       55174d7a7d46f29fc6be163ccb040330c357aa727b22ec13e2c5d713e5ba5b76"
    "F16       69e9a5adde6799bc1dbe52014f6fb35608d3763bac8bf482246ab1885c53d43c"
    "BF16      14b7a140ec1e91329e87508c259f18d2529d4f5adf2fe321bae1f99361e80dbb"
    "Q4_0      110a813b1e6a790e1721669c4deb526cf1a58c95546f1c0fdf7e54d2a37d7068"
    "Q4_1      a9ab77b87495d68882bafa6eea39f31596102cafaed03da7ffaaea8d7e7f61e3"
    "Q5_0      8163abe266b1a10d2be4f29496adc81a5372e418bf945b81184abed3ad965ffb"
    "Q5_1      c62221a9f8fddc79ac765909478801b7490f29a9ffaa9e7b328ed6187b47c6cd"
    "Q8_0      eb9dc6d93c6b5983ef2974e99ad3b06fd6514269a37b9674eef9cb8a66874344"
    "Q2_K      34822c88057a7bd4bfe91f3e7b99317a686c0193615993e9953b846819d056a2"
    "Q3_K      b6e433dc7e47dd0ea2f80111d6d69b368e4977787148d3d52041ec6b7a55d2c2"
    "Q4_K      ec6b1ff9a68d9353915f20b8f74330200b9ecdd9136b579602abb8bfeb5278b9"
    "Q5_K      5f9a07f75c0f5678a4a5e23654fae19287460fc941c4e960eccbe49bfec3a980"
    "Q6_K      5494d2f6b237e786a01287874d26261c1f21a3de812be4e65e2ca7a1f140f0af"
    "IQ4_NL    a40ac05e62aeaed04139f615ed7460ff4449d12a80bcde27889a213ba384264e"
    "IQ4_XS    c15ea97f4938d156cef3481a97247fb168b1594c7efafaa05b7fcd69f5edb63d"
    "MXFP4_MOE 66bffefcbaaff94f5291f03aab598623200511ab92bec2921ab47a9d568999b7"
)

check_sha256() {
    local name=$1
    local file=$2
    local expected=$3

    local actual
    actual=$(sha256sum "$file" | cut -d ' ' -f 1)

    local result=OK
    if [ "$no_check" = --no-check ]; then
        result=-
    elif [ "$actual" != "$expected" ]; then
        result=FAIL
    fi
    printf "%-16s %-6s %s\n" "$name" "$result" "$actual"

    if [ "$result" = FAIL ]; then
        echo "error: sha256 mismatch for $name: expected '$expected', got '$actual'" >&2
        exit 1
    fi
}

# keep the table readable, only show a command's output when it fails
run_logged() {
    local log=$1
    shift

    if ! "$@" > "$log" 2>&1; then
        cat "$log" >&2
        echo "error: command failed: $*" >&2
        exit 1
    fi
}

mkdir -p "$folder"

model=$folder/stories15M.gguf
if [ ! -f "$model" ]; then
    wget -q -O "$model" "$MODEL_URL"
fi

printf "%-16s %-6s %s\n" type result sha256
check_sha256 stories15M.gguf "$model" "$MODEL_SHA256"

f32=$folder/stories15M-be-F32.gguf
cp "$model" "$f32"
echo YES | run_logged "$folder/stories15M-be-F32.log" python3 "$(dirname "$0")/../gguf-py/gguf/scripts/gguf_convert_endian.py" "$f32" big

for entry in "${outputs[@]}"; do
    read -r type expected <<< "$entry"
    out=$folder/stories15M-be-$type.gguf

    if [ "$type" != F32 ]; then
        run_logged "$folder/stories15M-be-$type.log" "$quantize" --pure "$f32" "$out" "$type"
    fi

    check_sha256 "$type" "$out" "$expected"
done
