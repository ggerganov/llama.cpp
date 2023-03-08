#!/bin/sh

if [ ! -d deps ]
then
    mkdir deps
fi
cd deps
if [ ! -f v0.1.97.tar.gz ]
then
    curl -LO https://github.com/google/sentencepiece/archive/refs/tags/v0.1.97.tar.gz
fi
if [ ! -f libsentencepiece.a ]
then
    tar xzvf v0.1.97.tar.gz
    cd sentencepiece-0.1.97/ && rm -rf build && mkdir build && cd build
    cmake --version
    cmake ..
    make sentencepiece-static -j $(nproc)
    cd ../..
    cp sentencepiece-0.1.97/build/src/libsentencepiece.a ./
fi
cd ..
make
