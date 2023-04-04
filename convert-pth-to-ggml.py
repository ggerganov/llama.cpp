# Convert a LLaMA model checkpoint to a ggjt compatible file
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

QK = 32

GGML_TYPE_Q4_0  = 0
GGML_TYPE_Q4_1  = 1
GGML_TYPE_I8    = 2
GGML_TYPE_I16   = 3
GGML_TYPE_I32   = 4
GGML_TYPE_F16   = 5
GGML_TYPE_F32   = 6

WTYPES = {
    0: GGML_TYPE_F32,
    1: GGML_TYPE_F16,
    2: GGML_TYPE_Q4_0,
    3: GGML_TYPE_Q4_1,
}

GGML_BLCK_SIZE = {
    GGML_TYPE_Q4_0:  QK,
    GGML_TYPE_Q4_1:  QK,
    GGML_TYPE_I8:    1,
    GGML_TYPE_I16:   1,
    GGML_TYPE_I32:   1,
    GGML_TYPE_F16:   1,
    GGML_TYPE_F32:   1,
}

GGML_TYPE_SIZE = {
    GGML_TYPE_Q4_0: 4   + QK//2,
    GGML_TYPE_Q4_1: 4*2 + QK//2,
    GGML_TYPE_I8:   1,
    GGML_TYPE_I16:  2,
    GGML_TYPE_I32:  4,
    GGML_TYPE_F16:  2,
    GGML_TYPE_F32:  4,
}

def ggml_nelements(shape):
    r = 1
    for i in shape:
        r *= i
    return r

def ggml_nbytes(shape, ftype):
    x = ggml_nelements(shape)
    t = WTYPES[ftype]
    x *= GGML_TYPE_SIZE[t]
    x //= GGML_BLCK_SIZE[t]
    return x

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
        0x67676a74,  # magic: ggjt in hex
        1, # file version
        *[hparams[key] for key in keys],
        hparams["dim"] // hparams["n_heads"],  # rot (obsolete)
        ftype
    ]
    fout.write(struct.pack("i" * len(values), *values))

def write_tokens(fout, tokenizer):
    for i in range(tokenizer.vocab_size()):
        if tokenizer.is_unknown(i):
            text = " \u2047 ".encode()
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
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode()
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        fout.write(struct.pack("f", tokenizer.get_score(i)))

def process_and_write_variables(fout, model, ftype, part_id, n_parts):
    for name, datao in model.items():
        if name.endswith("freqs"):
            continue

        # remove dimensions with a single element
        data = datao.numpy().squeeze()
        partshape = data.shape
        n_dims = len(data.shape)
        assert n_dims in (1, 2)

        print(f"Processing variable: {name} with shape: {partshape} and type: {datao.dtype}")

        # coerce single-dimensional tensors from float16 to float32
        ftype_cur = 1
        if ftype == 0 or n_dims == 1:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
        blck_size = GGML_BLCK_SIZE[WTYPES[ftype_cur]]
        type_size = GGML_TYPE_SIZE[WTYPES[ftype_cur]]

        # determine dimension along which multipart tensor is sharded
        #
        # split_dim 0 regex:
        #   - output.*
        #   - layers.*.attention.wq.weight
        #   - layers.*.attention.wk.weight
        #   - layers.*.attention.wv.weight
        #   - layers.*.feed_forward.w1.weight
        #   - layers.*.feed_forward.w3.weight
        #
        # split_dim 1 regex:
        #   - tok_embeddings.*
        #   - layers.*.attention.wo.weight
        #   - layers.*.feed_forward.w2.weight
        #
        if n_dims > 1:
            split_dim = 1
            if "tok_embeddings" in name:
                split_dim = 1
            elif "layers" in name:
                if "attention.wo.weight" in name:
                    split_dim = 1
                elif "feed_forward.w2.weight" in name:
                    split_dim = 1
                else:
                    split_dim = 0
            elif "output" in name:
                split_dim = 0

        # output tensor header
        fullshape = list(partshape)
        if n_dims > 1:
            fullshape[split_dim] *= n_parts
        sname = name.encode()
        fout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
        for dim in reversed(fullshape):
            fout.write(struct.pack("i", dim))
        fout.write(sname)

        # ensure tensor data is aligned
        tensor_data_offset = fout.tell()
        while tensor_data_offset % QK != 0:
            fout.write(struct.pack("B", 0))
            tensor_data_offset += 1

        # output unified mappable tensor data
        if n_dims == 1 or n_parts == 1:
            # copy tensor which we thankfully received in one piece
            if part_id == 0:
                data.tofile(fout)
        elif split_dim == 0:
            # reassemble multifile tensor containing some of the rows
            rows_per_chunk = partshape[0]
            current_row = part_id * rows_per_chunk
            bytes_per_row = fullshape[1] // blck_size * type_size
            offset = current_row * bytes_per_row
            fout.seek(tensor_data_offset + offset)
            data.tofile(fout)
        elif split_dim == 1:
            # reassemble multifile tensor containing some of the cols
            cols_per_chunk = partshape[1]
            current_col = part_id * cols_per_chunk
            bytes_per_row = fullshape[1] // blck_size * type_size
            offset_current_col = current_col // blck_size * type_size
            for row in range(partshape[0]):
                offset_row = row * bytes_per_row
                offset = offset_row + offset_current_col
                fout.seek(tensor_data_offset + offset)
                data[row].tofile(fout)

        # advance file position to next tensor
        fout.seek(tensor_data_offset + ggml_nbytes(fullshape, ftype_cur))

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
        with open(fname_out, "wb") as fout:
            write_header(fout, hparams, ftype)
            write_tokens(fout, tokenizer)
        print(f"Done. Output file: {fname_out}\n")
        return

    n_parts = get_n_parts(hparams["dim"])
    fname_out = f"{dir_model}/ggml-model-{ftype_str[ftype]}.bin"

    # we output a single file for ggml
    with open(fname_out, "wb") as fout:
        write_header(fout, hparams, ftype)
        write_tokens(fout, tokenizer)
        offset_of_tensors = fout.tell()
        # the tensors we load could be split across multiple files
        for part_id in range(n_parts):
            fout.seek(offset_of_tensors)
            print(f"Processing part {part_id+1} of {n_parts}\n")
            fname_model = f"{dir_model}/consolidated.0{part_id}.pth"
            model = torch.load(fname_model, map_location="cpu")
            process_and_write_variables(fout, model, ftype, part_id, n_parts)
            del model

    print(f"Done. Output file: {fname_out}\n")

if __name__ == "__main__":
    main()
