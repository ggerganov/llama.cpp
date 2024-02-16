#!/bin/bash

set -e
set -x

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/compare-commits.sh <commit1> <commit2> [additional llama-bench arguments]"
    exit 1
fi

bench_args="${@:3}"

rm -f llama-bench.sqlite

git checkout $1
make clean && LLAMA_CUBLAS=1 make -j32 llama-bench
./llama-bench -o sql $bench_args | tee /dev/tty | sqlite3 llama-bench.sqlite

git checkout $2
make clean && LLAMA_CUBLAS=1 make -j32 llama-bench
./llama-bench -o sql $bench_args | tee /dev/tty | sqlite3 llama-bench.sqlite

./scripts/compare-llama-bench.py -b $1 -c $2
