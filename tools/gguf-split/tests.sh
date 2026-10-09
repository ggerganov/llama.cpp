#!/usr/bin/env bash

set -eu

if [ $# -lt 1 ]
then
    echo "usage:   $0 path_to_build_binary [path_to_temp_folder]"
    echo "example: $0 ../../build/bin ../../tmp"
    exit 1
fi

if [ $# -gt 1 ]
then
    TMP_DIR=$2
else
    TMP_DIR=/tmp
fi

set -x

SPLIT=$1/llama-gguf-split
MAIN=$1/llama-completion
WORK_PATH=$TMP_DIR/gguf-split
ROOT_DIR=$(realpath $(dirname $0)/../../)

mkdir -p "$WORK_PATH"

# Clean up in case of previously failed test
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-merge*.gguf

# 1. Get a model
(
cd $WORK_PATH
"$ROOT_DIR"/scripts/hf.sh --repo ggml-org/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q8_0.gguf
)
echo PASS

# 2. Split with max tensors strategy
$SPLIT --split-max-tensors 28  $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-split
echo PASS
echo

# 2b. Test the sharded model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-split-00001-of-00012.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 3. Merge
$SPLIT --merge $WORK_PATH/ggml-model-split-00001-of-00012.gguf $WORK_PATH/ggml-model-merge.gguf
echo PASS
echo

# 3b. Test the merged model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-merge.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 4. Split with no tensors in the first split
$SPLIT --split-max-tensors 32 --no-tensor-first-split $WORK_PATH/ggml-model-merge.gguf $WORK_PATH/ggml-model-split-32-tensors
echo PASS
echo

# 4b. Test the sharded model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-split-32-tensors-00001-of-00011.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 5. Merge
#$SPLIT --merge $WORK_PATH/ggml-model-split-32-tensors-00001-of-00011.gguf $WORK_PATH/ggml-model-merge-2.gguf
#echo PASS
#echo

# 5b. Test the merged model is loading properly
#$MAIN -no-cnv --model $WORK_PATH/ggml-model-merge-2.gguf -p "I believe the meaning of life is" --n-predict 32
#echo PASS
#echo

# 6. Split with size strategy
$SPLIT --split-max-size 500M $WORK_PATH/ggml-model-merge.gguf $WORK_PATH/ggml-model-split-500M
echo PASS
echo

# 6b. Test the sharded model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-split-500M-00001-of-00002.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 7. Merge with delete splits
#for i in $(seq -w 1 11); do
#    cp "$WORK_PATH/ggml-model-split-32-tensors-000${i}-of-00011.gguf" "$WORK_PATH/ggml-model-split-32-tensors-copy-000${i}-of-00011.gguf"
#done
#$SPLIT --merge --delete-splits $WORK_PATH/ggml-model-split-32-tensors-copy-00001-of-00011.gguf $WORK_PATH/ggml-model-merge-3.gguf
#echo PASS
#echo

# 7b. Test the merged model is loading properly
#$MAIN -no-cnv --model $WORK_PATH/ggml-model-merge-3.gguf -p "I believe the meaning of life is" --n-predict 32
#echo PASS
#echo

# 7c. Test the files were deleted
#for i in $(seq -w 1 11); do
#    test ! -f "$WORK_PATH/ggml-model-split-32-tensors-copy-000${i}-of-00011.gguf"
#done
#echo PASS
#echo

# 8. Regression test: a split/merge source file whose declared tensor data
#    extends past the actual end of the file (truncated, partially
#    downloaded, or otherwise corrupted) must be rejected with an error,
#    not silently "succeed" while writing stale/incorrect tensor bytes into
#    the output. This is a self-contained test (no model download needed).
python3 - "$WORK_PATH/offset-check-src.gguf" <<EOF
import sys
sys.path.insert(0, "$ROOT_DIR/gguf-py")
import gguf
import numpy as np

out_path = sys.argv[1]
w = gguf.GGUFWriter(out_path, "offset-check-test")
w.add_name("offset-check-test")
# a first tensor with a distinct, recognizable byte pattern, followed by a
# second, smaller tensor. Truncating the file right after the first
# tensor's data leaves the second tensor's declared offset/size pointing
# past EOF while its metadata (and the first tensor's leftover read buffer)
# remain intact -- the exact condition the offset bounds check must catch.
big = np.full(4096, 0xAA, dtype=np.uint8).view(np.float32)
small = np.full(16, 0xBB, dtype=np.uint8).view(np.float32)
w.add_tensor("tensor_big", big)
w.add_tensor("tensor_small", small)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
EOF
echo PASS
echo

# truncate right after tensor_big's data, so tensor_small's declared region
# lies entirely beyond the end of the file
TENSOR_SMALL_START=$(python3 - "$WORK_PATH/offset-check-src.gguf" <<EOF
import sys
sys.path.insert(0, "$ROOT_DIR/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
t = next(t for t in r.tensors if t.name == "tensor_small")
print(t.data_offset)
EOF
)
cp "$WORK_PATH/offset-check-src.gguf" "$WORK_PATH/offset-check-truncated.gguf"
truncate -s "$TENSOR_SMALL_START" "$WORK_PATH/offset-check-truncated.gguf"

# splitting the truncated file must FAIL (non-zero exit), not report success
if $SPLIT --split --split-max-tensors 1 "$WORK_PATH/offset-check-truncated.gguf" "$WORK_PATH/offset-check-split" 2>"$WORK_PATH/offset-check.err"; then
    echo "FAIL: gguf-split reported success on a truncated input file (expected a bounds-check error)"
    cat "$WORK_PATH/offset-check.err"
    exit 1
fi
grep -q "not within the file bounds" "$WORK_PATH/offset-check.err" || {
    echo "FAIL: gguf-split failed, but not with the expected bounds-check error"
    cat "$WORK_PATH/offset-check.err"
    exit 1
}
echo PASS
echo

# sanity: splitting the original, untruncated file must still succeed, and
# the split shard's tensor_small bytes must match the real 0xBB pattern,
# not leftover 0xAA bytes from tensor_big
$SPLIT --split --split-max-tensors 1 "$WORK_PATH/offset-check-src.gguf" "$WORK_PATH/offset-check-valid-split"
python3 - "$WORK_PATH/offset-check-valid-split-00002-of-00002.gguf" <<EOF
import sys
sys.path.insert(0, "$ROOT_DIR/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
t = next(t for t in r.tensors if t.name == "tensor_small")
data = t.data.tobytes()
assert all(b == 0xBB for b in data), f"expected all 0xBB bytes, got: {data.hex()}"
EOF
echo PASS
echo

# Clean up
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-merge*.gguf
rm -f $WORK_PATH/offset-check*.gguf $WORK_PATH/offset-check.err
