#!/bin/bash
cd `dirname $0`
cd ../..

EXE="./finetune"

MODEL="openllama-3b-v2.gguf"

while getopts "dg" opt; do
  case $opt in
    d)
      DEBUGGER="gdb --args"
      ;;
    g)
      # GPU. The makefile doesn't support CUDA on Windows, so I have to use CMake and so main is built to a different location.
      # Note: "-d" doesn't really work with this - it will run under gdb, but there are no debugging symbols (in a format gdb understands). I think the easiest workaround is to use WinDbg instead.
      EXE="./build/bin/Release/finetune"
      GPUARG="--gpu-layers 25"
      ;;
  esac
done

$DEBUGGER $EXE \
        --model-base c:/models/$MODEL \
	$GPUARG \
        --checkpoint-in  chk-ol3b-shakespeare-LATEST.gguf \
        --checkpoint-out chk-ol3b-shakespeare-ITERATION.gguf \
        --lora-out lora-ol3b-shakespeare-ITERATION.bin \
        --train-data "c:\training\shakespeare.txt" \
        --save-every 10 \
        --threads 10 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing
