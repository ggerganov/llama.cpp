#!/bin/sh

BUILD_NUMBER="0"
BUILD_COMMIT="unknown"

REV_LIST=$(git rev-list --count HEAD)
if [ $? -eq 0 ]; then
  BUILD_NUMBER=$REV_LIST
fi

REV_PARSE=$(git rev-parse --short HEAD)
if [ $? -eq 0 ]; then
  BUILD_COMMIT=$REV_PARSE
fi

echo "#ifndef BUILD_INFO_H"
echo "#define BUILD_INFO_H"
echo ""
echo "#define BUILD_NUMBER $BUILD_NUMBER"
echo "#define BUILD_COMMIT \"$BUILD_COMMIT\""
echo ""
echo "#endif // BUILD_INFO_H"
