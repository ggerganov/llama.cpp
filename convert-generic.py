#!/usr/bin/env python3

from __future__ import annotations
from util import parse_args

import sys
import model


args = parse_args()

dir_model = args.model
ftype = args.ftype
if not dir_model.is_dir():
    print(f'Error: {args.model} is not a directory', file=sys.stderr)
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

print(f"Loading model: {dir_model.name}")

hparams = model.Model.load_hparams(dir_model)

model_class = model.Model.from_model_architecture(hparams["architectures"][0])
model_instance = model_class(dir_model, ftype, fname_out)

print("Set model parameters")
model_instance.set_gguf_parameters()

print("Set model tokenizer")
model_instance.set_vocab()

if not args.vocab_only:
    print(f"Exporting model to '{fname_out}'")
    model_instance.write()
else:
    print(f"Exporting model vocab to '{fname_out}'")
    model_instance.write_vocab()

print(f"Model successfully exported to '{fname_out}'")
print("")
