# Migrate ggml file(s) with ggmf magic to ggml file with ggjt magic
#
# We caused a breaking change to the file format on 2023-03-30 in:
#     https://github.com/ggerganov/llama.cpp/pull/613
#
# (1) If you still have the Meta LLaMA .pth files, then close this
#     file now; you can just run `convert-pth-to-ggml.py` again to
#     migrate to the new format. The tool is easier to use too. It
#     isn't necessary anymore to manage split output files because
#     the new format always combines things into a single file.
#
# (2) If you deleted the Meta LLaMA .pth files due to save on disk
#     space, then this tool is intended to help you.  Please check
#     out the instructions below.
#
# USAGE
#
#     python migrate-ggml-2023-03-30-pr613.py INPUT OUTPUT
#
# PREREQUISITES
#
#     pip install numpy
#     cd llama.cpp
#     make -j4
#
# EXAMPLE (7B MODEL)
#
#     # you can replace all the 'f16' with 'q4_0' if you're using quantized weights
#     python migrate-ggml-2023-03-30-pr613.py models/7B/ggml-model-f16.bin models/7B/ggml-model-f16-ggjt.bin
#
#     # check that it works
#     ./main -m models/7B/ggml-model-f16-ggjt.bin -p 'Question: Do you love me?'
#
#     # you can delete the old files
#     rm -f models/7B/ggml-model-f16.bin
#     mv models/7B/ggml-model-f16-ggjt.bin models/7B/ggml-model-f16.bin
#
# EXAMPLE (13B MODEL)
#
#     # you can replace all the 'f16' with 'q4_0' if you're using quantized weights
#     python migrate-ggml-2023-03-30-pr613.py models/13B/ggml-model-f16.bin models/13B/ggml-model-f16-ggjt.bin
#
#     # check that it works
#     ./main -m models/13B/ggml-model-f16-ggjt.bin -p 'Question: Do you love me?'
#
#     # you can delete the old files
#     rm -f models/13B/ggml-model-f16.bin*
#     mv models/13B/ggml-model-f16-ggjt.bin models/13B/ggml-model-f16.bin
#

import argparse
import os
import sys
import json
import struct
import numpy as np

QK = 32

GGML_TYPE_Q4_0  = 0
GGML_TYPE_Q4_1  = 1
GGML_TYPE_I8    = 2
GGML_TYPE_I16   = 3
GGML_TYPE_I32   = 4
GGML_TYPE_F16   = 5
GGML_TYPE_F32   = 6

WTYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
}

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

HPARAMS = [
    'magic',    # int32
    'version',  # int32
    'n_vocab',  # int32
    'n_embd',   # int32
    'n_mult',   # int32
    'n_head',   # int32
    'n_layer',  # int32
    'n_rot',    # int32
    'f16',      # int32
]

def read_hparams(fin):
    struct_fmt = "i" * len(HPARAMS)
    struct_size = struct.calcsize(struct_fmt)
    buf = fin.read(struct_size)
    ints = struct.unpack(struct_fmt, buf)
    hparams = dict(zip(HPARAMS, ints))
    return hparams

def write_hparams(fout, hparams):
    struct_fmt = "i" * len(HPARAMS)
    struct_size = struct.calcsize(struct_fmt)
    ints = [hparams[h] for h in HPARAMS]
    fout.write(struct.pack(struct_fmt, *ints))

def read_tokens(fin, hparams):
    tokens = []
    for i in range(hparams['n_vocab']):
        len_b = fin.read(4)
        (length,) = struct.unpack("i", len_b)
        word = fin.read(length)
        score_b = fin.read(4)
        (score,) = struct.unpack("f", score_b)
        tokens.append((word, score))
    return tokens

def write_tokens(fout, tokens):
    for word, score in tokens:
        fout.write(struct.pack("i", len(word)))
        fout.write(word)
        fout.write(struct.pack("f", score))

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

def copy_tensors(fin, fout, part_id, n_parts):
    while True:

        b = fin.read(4)
        if not b: break
        (n_dims,) = struct.unpack("i", b)
        b = fin.read(4)
        (length,) = struct.unpack("i", b)
        b = fin.read(4)
        (ftype,) = struct.unpack("i", b)

        assert n_dims in (1, 2)

        partshape = list(range(n_dims))
        for i in range(n_dims):
            b = fin.read(4)
            partshape[i] = struct.unpack("i", b)[0]
        partshape = list(reversed(partshape))

        name = fin.read(length)
        data = fin.read(ggml_nbytes(partshape, ftype))

        blck_size = GGML_BLCK_SIZE[WTYPES[ftype]]
        type_size = GGML_TYPE_SIZE[WTYPES[ftype]]

        print(f"Processing tensor {name} with shape: {partshape} and type: {WTYPE_NAMES[ftype]}")

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
            if b"tok_embeddings" in name:
                split_dim = 1
            elif b"layers" in name:
                if b"attention.wo.weight" in name:
                    split_dim = 1
                elif b"feed_forward.w2.weight" in name:
                    split_dim = 1
                else:
                    split_dim = 0
            elif b"output" in name:
                split_dim = 0

        # output tensor header
        fullshape = list(partshape)
        if n_dims > 1:
            fullshape[split_dim] *= n_parts
        fout.write(struct.pack("iii", n_dims, len(name), ftype))
        for dim in reversed(fullshape):
            fout.write(struct.pack("i", dim))
        fout.write(name)

        # ensure tensor data is aligned
        tensor_data_offset = fout.tell()
        while tensor_data_offset % QK != 0:
            fout.write(struct.pack("B", 0))
            tensor_data_offset += 1

        # output unified mappable tensor data
        if n_dims == 1 or n_parts == 1:
            # copy tensor which we thankfully received in one piece
            if part_id == 0:
                fout.write(data)
        elif split_dim == 0:
            # reassemble multifile tensor containing some of the rows
            rows_per_chunk = partshape[0]
            current_row = part_id * rows_per_chunk
            bytes_per_row = fullshape[1] // blck_size * type_size
            offset = current_row * bytes_per_row
            fout.seek(tensor_data_offset + offset)
            fout.write(data)
        elif split_dim == 1:
            # reassemble multifile tensor containing some of the cols
            cols_per_chunk = partshape[1]
            current_col = part_id * cols_per_chunk
            bpr = partshape[1] // blck_size * type_size
            bytes_per_row = fullshape[1] // blck_size * type_size
            offset_current_col = current_col // blck_size * type_size
            for row in range(partshape[0]):
                offset_row = row * bytes_per_row
                offset = offset_row + offset_current_col
                fout.seek(tensor_data_offset + offset)
                fout.write(data[row * bpr:row * bpr + bpr])

        # advance file position to next tensor
        fout.seek(tensor_data_offset + ggml_nbytes(fullshape, ftype))

def parse_args():
    parser = argparse.ArgumentParser(description='Migrate from GGML to new GGJT file format')
    parser.add_argument('fin_path', help='your old ggml file (leave out the .1 .2 etc.)')
    parser.add_argument('fout_path', help='your new ggjt file name')
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.fin_path
    assert args.fout_path
    assert args.fin_path != args.fout_path

    with open(args.fin_path, "rb") as fin:
        hparams = read_hparams(fin)
        tokens = read_tokens(fin, hparams)

    if hparams['magic'] == 0x67676a74:  # ggjt
        print(f"{args.fin_path}: input ggml has already been converted to 'ggjt' magic\n")
        sys.exit(1)

    if hparams['magic'] != 0x67676d66:  # ggmf
        print(f"{args.fin_path}: input ggml file doesn't have expected 'ggmf' magic: {hparams['magic']:#x}\n")
        sys.exit(1)

    hparams['magic'] = 0x67676a74  # ggjt

    # count number of multipart files by convention
    n_parts = 1
    while True:
        if os.path.exists(f"{args.fin_path}.{n_parts}"):
            n_parts += 1
        else:
            break

    # we output a single file for ggml
    with open(args.fout_path, "wb") as fout:
        write_hparams(fout, hparams)
        write_tokens(fout, tokens)
        offset_of_tensors = fout.tell()
        # the tensors we load could be split across multiple files
        for part_id in range(n_parts):
            fout.seek(offset_of_tensors)
            print(f"Processing part {part_id+1} of {n_parts}\n")
            fin_path = args.fin_path
            if part_id > 0:
                fin_path += f".{part_id}"
            with open(fin_path, "rb") as fin:
                read_tokens(fin, read_hparams(fin))
                copy_tensors(fin, fout, part_id, n_parts)

    print(f"Done. Output file: {args.fout_path}\n")

if __name__ == "__main__":
    main()
