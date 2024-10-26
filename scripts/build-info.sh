#!/bin/sh

CC=$1

build_number="0"
build_commit="unknown"
build_compiler="unknown"
build_target="unknown"

if out=$(git rev-list --count HEAD); then
    # git is broken on WSL so we need to strip extra newlines
    build_number=$(printf '%s' "$out" | tr -d '\n')
fi

if out=$(git rev-parse --short HEAD); then
    build_commit=$(printf '%s' "$out" | tr -d '\n')
fi

if out=$($CC --version | head -1); then
    build_compiler=$out
fi

if out=$($CC -dumpmachine); then
    build_target=$out
fi

echo "int JARVIS_BUILD_NUMBER = ${build_number};"
echo "char const *JARVIS_COMMIT = \"${build_commit}\";"
echo "char const *JARVIS_COMPILER = \"${build_compiler}\";"
echo "char const *JARVIS_BUILD_TARGET = \"${build_target}\";"
