#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/compare-commits.sh <commit1> <commit2> [additional jarvis-bench arguments]"
    exit 1
fi

set -e
set -x

# verify at the start that the compare script has all the necessary dependencies installed
./scripts/compare-jarvis-bench.py --check

bench_args="${@:3}"

rm -f jarvis-bench.sqlite > /dev/null

# to test a backend, call the script with the corresponding environment variable (e.g. GGML_CUDA=1 ./scripts/compare-commits.sh ...)

git checkout $1 > /dev/null
make clean > /dev/null
make -j$(nproc) $make_opts jarvis-bench > /dev/null
./jarvis-bench -o sql -oe md $bench_args | sqlite3 jarvis-bench.sqlite

git checkout $2 > /dev/null
make clean > /dev/null
make -j$(nproc) $make_opts jarvis-bench > /dev/null
./jarvis-bench -o sql -oe md $bench_args | sqlite3 jarvis-bench.sqlite

./scripts/compare-jarvis-bench.py -b $1 -c $2
