#!/bin/env python3

import argparse
import os
import subprocess as sp
from glob import glob

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='command')

parser.add_argument(
    '-m', "--model", type=str, required=True,
    help="Directory containing model file, or model file itself (*.pth, *.pt, *.bin)")

run = subparsers.add_parser("run", help="Run a model previously converted into ggml")
convert = subparsers.add_parser("convert", help="Convert a llama model into ggml")
quantize = subparsers.add_parser("quantize", help="Optimize with quantization process ggml")
allinone = subparsers.add_parser("all-in-one", help="Execute --convert & --quantize")
server = subparsers.add_parser("server", help="Execute in server mode ex: -m /models/7B/ggml-model-q4_0.bin -c 2048 -ngl 43 -mg 1 --port 8080")

known_args, unknown_args = parser.parse_known_args()
model_path = known_args.model
converted_models = glob(os.path.join(model_path, 'ggml-model-*.gguf'))

if known_args.command == 'convert':
    sp.run(['python3', './convert.py', model_path] + unknown_args, check=True)

if known_args.command == 'run':
    sp.run(['./main', '-m', model_path] + unknown_args, check=True)

if known_args.command == 'quantize':
    if not converted_models:
        print(f"No models ready for quantization found in {model_path}")
        exit(1)
    sp.run(['./quantize', converted_models[0]] + unknown_args, check=True)

if known_args.command == 'all-in-one':
    if not converted_models:
        sp.run(['python3', './convert.py', model_path], check=True)
        converted_models = glob(os.path.join(model_path, 'ggml-model-*.gguf'))
    else:
        print(
            f"Converted models found {converted_models}! No need to convert.")

    quantized_models = glob(os.path.join(model_path, f'ggml-model-q*_*.bin'))

    if not quantized_models:
        sp.run(['./quantize', converted_models[0]] + unknown_args, check=True)
    else:
        print(
            f"Quantized models found {quantized_models}! No need to quantize.")
if known_args.command == "server":
   sp.run(['./server', '-m', model_path] + unknown_args, check=True)

exit()
