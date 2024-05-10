#!/bin/bash
#
# Synchronize ggml changes to llama.cpp
#
# Usage:
#
#   $ cd /path/to/llama.cpp
#   $ ./scripts/sync-ggml-am.sh -skip hash0,hash1,hash2...
#

set -e

sd=$(dirname $0)
cd $sd/../

SRC_LLAMA=$(pwd)
SRC_GGML=$(cd ../ggml; pwd)

if [ ! -d $SRC_GGML ]; then
    echo "ggml not found at $SRC_GGML"
    exit 1
fi

lc=$(cat $SRC_LLAMA/scripts/sync-ggml.last)
echo "Syncing ggml changes since commit $lc"

to_skip=""
if [ "$1" == "-skip" ]; then
    to_skip=$2
fi

cd $SRC_GGML

git log --oneline $lc..HEAD
git log --oneline $lc..HEAD --reverse | grep -v "(llama/[0-9]*)" | cut -d' ' -f1 > $SRC_LLAMA/ggml-commits

if [ ! -s $SRC_LLAMA/ggml-commits ]; then
    rm -v $SRC_LLAMA/ggml-commits
    echo "No new commits"
    exit 0
fi

if [ -f $SRC_LLAMA/ggml-src.patch ]; then
    rm -v $SRC_LLAMA/ggml-src.patch
fi

while read c; do
    if [ -n "$to_skip" ]; then
        if [[ $to_skip == *"$c"* ]]; then
            echo "Skipping $c"
            continue
        fi
    fi

    git format-patch -k $c~1..$c --stdout -- \
        include/ggml/ggml*.h \
        src/ggml*.h \
        src/ggml*.c \
        src/ggml*.cpp \
        src/ggml*.m \
        src/ggml*.metal \
        src/ggml*.cu \
        src/ggml-cuda/* \
        tests/test-opt.cpp \
        tests/test-grad0.cpp \
        tests/test-quantize-fns.cpp \
        tests/test-quantize-perf.cpp \
        tests/test-backend-ops.cpp \
        LICENSE \
        scripts/gen-authors.sh \
        >> $SRC_LLAMA/ggml-src.patch
done < $SRC_LLAMA/ggml-commits

rm -v $SRC_LLAMA/ggml-commits

# delete files if empty
if [ ! -s $SRC_LLAMA/ggml-src.patch ]; then
    rm -v $SRC_LLAMA/ggml-src.patch
fi

cd $SRC_LLAMA

if [ -f $SRC_LLAMA/ggml-src.patch ]; then
    # replace PR numbers
    #
    # Subject: some text (#1234)
    # Subject: some text (ggml/1234)
    cat ggml-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    cat ggml-src.patch | sed -e 's/^\(.*\) (#\([0-9]*\))$/\1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    # replace filenames:
    #
    # src/ggml.c                  -> ggml.c
    # src/ggml-alloc.c            -> ggml-alloc.c
    # src/ggml-backend-impl.h     -> ggml-backend-impl.h
    # src/ggml-backend.c          -> ggml-backend.c
    # src/ggml-common.h           -> ggml-common.h
    # src/ggml-cuda/*             -> ggml-cuda/
    # src/ggml-cuda.cu            -> ggml-cuda.cu
    # src/ggml-cuda.h             -> ggml-cuda.h
    # src/ggml-impl.h             -> ggml-impl.h
    # src/ggml-kompute.cpp        -> ggml-kompute.cpp
    # src/ggml-kompute.h          -> ggml-kompute.h
    # src/ggml-metal.h            -> ggml-metal.h
    # src/ggml-metal.m            -> ggml-metal.m
    # src/ggml-mpi.h              -> ggml-mpi.h
    # src/ggml-mpi.c              -> ggml-mpi.c
    # src/ggml-opencl.cpp         -> ggml-opencl.cpp
    # src/ggml-opencl.h           -> ggml-opencl.h
    # src/ggml-quants.c           -> ggml-quants.c
    # src/ggml-quants.h           -> ggml-quants.h
    # src/ggml-sycl.cpp           -> ggml-sycl.cpp
    # src/ggml-sycl.h             -> ggml-sycl.h
    # src/ggml-vulkan.cpp         -> ggml-vulkan.cpp
    # src/ggml-vulkan.h           -> ggml-vulkan.h
    # include/ggml/ggml.h         -> ggml.h
    # include/ggml/ggml-alloc.h   -> ggml-alloc.h
    # include/ggml/ggml-backend.h -> ggml-backend.h
    #
    # tests/test-opt.cpp           -> tests/test-opt.cpp
    # tests/test-grad0.cpp         -> tests/test-grad0.cpp
    # tests/test-quantize-fns.cpp  -> tests/test-quantize-fns.cpp
    # tests/test-quantize-perf.cpp -> tests/test-quantize-perf.cpp
    # tests/test-backend-ops.cpp   -> tests/test-backend-ops.cpp
    #
    # LICENSE                      -> LICENSE
    # scripts/gen-authors.sh       -> scripts/gen-authors.sh

    cat ggml-src.patch | sed \
        -e 's/src\/ggml\.c/ggml.c/g' \
        -e 's/src\/ggml-alloc\.c/ggml-alloc.c/g' \
        -e 's/src\/ggml-backend-impl\.h/ggml-backend-impl.h/g' \
        -e 's/src\/ggml-backend\.c/ggml-backend.c/g' \
        -e 's/src\/ggml-common\.h/ggml-common.h/g' \
        -e 's/src\/ggml-cuda\//ggml-cuda\//g' \
        -e 's/src\/ggml-cuda\.cu/ggml-cuda.cu/g' \
        -e 's/src\/ggml-cuda\.h/ggml-cuda.h/g' \
        -e 's/src\/ggml-impl\.h/ggml-impl.h/g' \
        -e 's/src\/ggml-kompute\.cpp/ggml-kompute.cpp/g' \
        -e 's/src\/ggml-kompute\.h/ggml-kompute.h/g' \
        -e 's/src\/ggml-metal\.h/ggml-metal.h/g' \
        -e 's/src\/ggml-metal\.m/ggml-metal.m/g' \
        -e 's/src\/ggml-mpi\.h/ggml-mpi.h/g' \
        -e 's/src\/ggml-mpi\.c/ggml-mpi.c/g' \
        -e 's/src\/ggml-opencl\.cpp/ggml-opencl.cpp/g' \
        -e 's/src\/ggml-opencl\.h/ggml-opencl.h/g' \
        -e 's/src\/ggml-quants\.c/ggml-quants.c/g' \
        -e 's/src\/ggml-quants\.h/ggml-quants.h/g' \
        -e 's/src\/ggml-sycl\.cpp/ggml-sycl.cpp/g' \
        -e 's/src\/ggml-sycl\.h/ggml-sycl.h/g' \
        -e 's/src\/ggml-vulkan\.cpp/ggml-vulkan.cpp/g' \
        -e 's/src\/ggml-vulkan\.h/ggml-vulkan.h/g' \
        -e 's/include\/ggml\/ggml\.h/ggml.h/g' \
        -e 's/include\/ggml\/ggml-alloc\.h/ggml-alloc.h/g' \
        -e 's/include\/ggml\/ggml-backend\.h/ggml-backend.h/g' \
        -e 's/tests\/test-opt\.cpp/tests\/test-opt.cpp/g' \
        -e 's/tests\/test-grad0\.cpp/tests\/test-grad0.cpp/g' \
        -e 's/tests\/test-quantize-fns\.cpp/tests\/test-quantize-fns.cpp/g' \
        -e 's/tests\/test-quantize-perf\.cpp/tests\/test-quantize-perf.cpp/g' \
        -e 's/tests\/test-backend-ops\.cpp/tests\/test-backend-ops.cpp/g' \
        -e 's/LICENSE/LICENSE/g' \
        -e 's/scripts\/gen-authors\.sh/scripts\/gen-authors.sh/g' \
        > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    git am ggml-src.patch

    rm -v $SRC_LLAMA/ggml-src.patch
fi

# update last commit
cd $SRC_GGML
git log -1 --format=%H > $SRC_LLAMA/scripts/sync-ggml.last

echo "Done"

exit 0
