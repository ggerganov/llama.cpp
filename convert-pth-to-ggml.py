# Convert a LLaMA model checkpoint to a ggml compatible file
#
# Load the model using Torch
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import argparse
import os
import sys
import json
import struct
import numpy as np
import torch

from sentencepiece import SentencePieceProcessor

def parse_args():

    parser = argparse.ArgumentParser(description='Convert a LLaMA model checkpoint to a ggml compatible file')
    parser.add_argument('dir_model',  help='directory containing the model checkpoint')
    parser.add_argument('ftype',      help='file type (0: float32, 1: float16)', type=int, choices=[0, 1], default=1)
    parser.add_argument('vocab_only', help='only write vocab to file', type=int, default=0, nargs='?')
    return parser.parse_args()

def get_n_parts(dim):

    mappings = {4096: 1, 5120: 2, 6656: 4, 8192: 8}
    n_parts = mappings.get(dim)
    if n_parts is None:
        print(f"Invalid dim: {dim}")
        sys.exit(1)

    print(f"n_parts = {n_parts}\n")
    return n_parts

def load_hparams_and_tokenizer(dir_model):

    # `dir_model` is something like `models/7B` or `models/7B/`.
    # "tokenizer.model" is expected under model's parent dir.
    # When `dir_model` is a symlink, f"{dir_model}/../tokenizer.model" would not be found.
    # Let's use the model's parent dir directly.
    model_parent_dir = os.path.dirname(os.path.normpath(dir_model))

    fname_hparams = f"{dir_model}/params.json"
    fname_tokenizer = f"{model_parent_dir}/tokenizer.model"

    with open(fname_hparams, "r") as f:
        hparams = json.load(f)
        print(hparams)

    tokenizer = SentencePieceProcessor(fname_tokenizer)
    hparams.update({"vocab_size": tokenizer.vocab_size()})

    return hparams, tokenizer

def write_header(fout, hparams, ftype):

    keys = ["vocab_size", "dim", "multiple_of", "n_heads", "n_layers"]
    values = [
        0x67676d66,  # magic: ggmf in hex
        1, # file version
        *[hparams[key] for key in keys],
        hparams["dim"] // hparams["n_heads"],  # rot (obsolete)
        ftype
    ]
    fout.write(struct.pack("i" * len(values), *values))

def write_tokens(fout, tokenizer):

    for i in range(tokenizer.vocab_size()):
        if tokenizer.is_unknown(i):
            text = " \u2047 ".encode("utf-8")
        elif tokenizer.is_control(i):
            text = b""
        elif tokenizer.is_byte(i):
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                print(f"Invalid token: {piece}")
                sys.exit(1)
            byte_value = int(piece[3:-1], 16)
            text = struct.pack("B", byte_value)
        else:
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        fout.write(struct.pack("f", tokenizer.get_score(i)))

def process_and_write_variables(fout, model, ftype):

    for name, datao in model.items():

        if name.endswith("freqs"):
            continue

        shape = datao.shape

        print(f"Processing variable: {name} with shape: {shape} and type: {datao.dtype}")

        data = datao.numpy().squeeze()
        n_dims = len(shape)

        # default type is fp16
        ftype_cur = 1
        if ftype == 0 or n_dims == 1:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

        # header
        sname = name.encode('utf-8')
        fout.write(struct.pack("iii", len(data.shape), len(sname), ftype_cur))
        for dim in reversed(data.shape):
            fout.write(struct.pack("i", dim))
        fout.write(sname)

        # data output to file
        data.tofile(fout)

def main():

    args = parse_args()
    dir_model = args.dir_model
    ftype = args.ftype
    ftype_str = ["f32", "f16"]

    hparams, tokenizer = load_hparams_and_tokenizer(dir_model)

    print(args)

    # if only writing vocab to file
    if args.vocab_only:

        fname_model = f"{dir_model}/consolidated.00.pth"
        fname_out = f"{dir_model}/ggml-vocab.bin"

        print(f"Extracting only the vocab from '{fname_model}'\n")

        model = torch.load(fname_model, map_location="cpu")

        with open(fname_out, "wb") as fout:
            write_header(fout, hparams, ftype)
            write_tokens(fout, tokenizer)

        del model

        print(f"Done. Output file: {fname_out}\n")

        return

    n_parts = get_n_parts(hparams["dim"])

    for p in range(n_parts):

        print(f"Processing part {p}\n")

        fname_model = f"{dir_model}/consolidated.0{p}.pth"
        fname_out = f"{dir_model}/ggml-model-{ftype_str[ftype]}.bin{'' if p == 0 else '.' + str(p)}"

        model = torch.load(fname_model, map_location="cpu")

        with open(fname_out, "wb") as fout:
            write_header(fout, hparams, ftype)
            write_tokens(fout, tokenizer)
            process_and_write_variables(fout, model, ftype)

        del model

        print(f"Done. Output file: {fname_out}, (part {p})\n")

if __name__ == "__main__":
    main()
