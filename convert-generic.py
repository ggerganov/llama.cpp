#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
   sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))

import gguf
import model
import util

args = util.parse_args()

dir_model = args.model
ftype = args.ftype
if not dir_model.is_dir():
    print(f'Error: {args.model} is not a directory', file = sys.stderr)
    sys.exit(1)

# possible tensor data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16

# map from ftype to string
ftype_str = ["f32", "f16"]

if args.outfile is not None:
    fname_out = args.outfile
else:
    # output in the same directory as the model by default
    fname_out = dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'


print("gguf: loading model " + dir_model.name)

hparams = model.Model.load_hparams(dir_model)

model_class = model.Model.from_model_architecture(hparams["architectures"][0])
model_instance = model_class(dir_model, ftype, fname_out)

print("gguf: get model metadata")

model_instance.set_gguf_parameters()

# TOKENIZATION
print("gguf: get tokenizer metadata")

print("gguf: get gpt2 tokenizer vocab")

model_instance.set_vocab()

# write model
print("gguf: write header")
model_instance.gguf_writer.write_header_to_file()
print("gguf: write metadata")
model_instance.gguf_writer.write_kv_data_to_file()
if not args.vocab_only:
    print("gguf: write tensors")
    model_instance.write_tensors()
    model_instance.gguf_writer.write_tensors_to_file()

model_instance.gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
