#!/usr/bin/env python3

#
# TODO: deduplicate GPT4All with convert-unversioned-ggml-to-ggml.py
#

# Original by https://github.com/eiz
# https://github.com/ggerganov/llama.cpp/issues/324#issuecomment-1476227818
import argparse
import glob
import os
import struct
import sys
from sentencepiece import SentencePieceProcessor

HPARAMS = keys = ["vocab_size", "dim", "multiple_of", "n_heads", "n_layers"]

def parse_args():
    parser = argparse.ArgumentParser(description='Upgrade a GPT4All model to the current format')
    parser.add_argument('gpt4all_model', help='path to gpt4all-lora-quantized.bin')
    parser.add_argument('tokenizer_model', help='path to LLaMA tokenizer.model file')
    return parser.parse_args()

def read_header(f_in):
    struct_fmt = "i" * (3 + len(HPARAMS))
    struct_size = struct.calcsize(struct_fmt)
    buf = f_in.read(struct_size)
    return struct.unpack(struct_fmt, buf)

def write_header(f_out, header):
    (magic, vocab_size, dim, multiple_of, n_heads, n_layers, rot, ftype) = header

    if magic != 0x67676d6c:
        raise Exception('Invalid file magic. Must be an old style ggml file.')

    values = [
        0x67676d66, # magic: ggml in hex
        1,          # file version
        vocab_size,
        dim,
        multiple_of,
        n_heads,
        n_layers,
        rot,
        ftype
    ]
    f_out.write(struct.pack("i" * len(values), *values))

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

    # TODO: GPT4All - add extra <pad> token
    text = "<pad>".encode()
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    fout.write(struct.pack("f", 0.0))

def read_tokens(f_in, tokenizer):
    for i in range(tokenizer.vocab_size()):
        len_b = f_in.read(4)
        (length,) = struct.unpack("i", len_b)
        f_in.read(length)

def copy_all_data(f_out, f_in):
    while True:
        buf = f_in.read(1024 * 1024)
        if not buf:
            break
        f_out.write(buf)

def convert_one_file(path_in, tokenizer):
    path_tmp = f"{path_in}.tmp"
    path_orig= f"{path_in}.orig"
    print(f"converting {path_in}")
    with open(path_in, "rb") as f_in, open(path_tmp, "wb") as f_out:
        write_header(f_out, read_header(f_in))
        read_tokens(f_in, tokenizer)
        write_tokens(f_out, tokenizer)
        copy_all_data(f_out, f_in)
    os.rename(path_in, path_orig)
    os.rename(path_tmp, path_in)

def main():
    args = parse_args()

    tokenizer = SentencePieceProcessor(args.tokenizer_model)

    convert_one_file(args.gpt4all_model, tokenizer)

if __name__ == "__main__":
    main()
