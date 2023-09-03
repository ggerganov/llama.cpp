#!/bin/sh

build_number="0"
build_commit="unknown"

# git is broken on WSL so we need to strip extra newlines
if out=$(git rev-list --count HEAD | tr -d '\n'); then
  build_number=$out
fi

if out=$(git rev-parse --short HEAD | tr -d '\n'); then
  build_commit=$out
fi

echo "#ifndef BUILD_INFO_H"
echo "#define BUILD_INFO_H"
echo
echo "#define BUILD_NUMBER $build_number"
echo "#define BUILD_COMMIT \"$build_commit\""
echo
echo "#endif // BUILD_INFO_H"
