#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/compare-commits.sh <commit1> <commit2> [additional llama-bench arguments]"
    exit 1
fi

set -e
set -x

bench_args="${@:3}"

rm -f llama-bench.sqlite

backend="cpu"

if [[ "$OSTYPE" == "darwin"* ]]; then
    backend="metal"
elif command -v nvcc &> /dev/null; then
    backend="cuda"
fi

make_opts=""

if [[ "$backend" == "cuda" ]]; then
    make_opts="LLAMA_CUBLAS=1"
fi

git checkout $1
make clean && make -j32 $make_opts llama-bench
./llama-bench -o sql $bench_args | tee /dev/tty | sqlite3 llama-bench.sqlite

git checkout $2
make clean && make -j32 $make_opts llama-bench
./llama-bench -o sql $bench_args | tee /dev/tty | sqlite3 llama-bench.sqlite

./scripts/compare-llama-bench.py -b $1 -c $2
