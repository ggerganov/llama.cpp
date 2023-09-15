#!/usr/bin/env python3
# HF starcoder --> gguf conversion

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore[import]

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf


def bytes_to_unicode():
    # ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


def count_model_parts(dir_model: Path) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a StarCoder model to a GGML compatible file")
    parser.add_argument("--vocab-only", action="store_true", help="extract only the vocab")
    parser.add_argument("--outfile",    type=Path,           help="path to write to; default: based on input")
    parser.add_argument("model",        type=Path,           help="directory containing model file, or model file itself (*.bin)")
    parser.add_argument("ftype",        type=int,            help="output format - use 0 for float32, 1 for float16", choices=[0, 1], default = 1)
    return parser.parse_args()

args = parse_args()

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

print("gguf: loading model "+dir_model.name)

with open(dir_model / "config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

if hparams["architectures"][0] != "GPTBigCodeForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit(1)

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.STARCODER
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["n_layer"]

gguf_writer.add_name("StarCoder")
gguf_writer.add_context_length(hparams["n_positions"])
gguf_writer.add_embedding_length(hparams["n_embd"])
gguf_writer.add_feed_forward_length(4 * hparams["n_embd"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_head_count(hparams["n_head"])
gguf_writer.add_head_count_kv(1)
gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])
gguf_writer.add_file_type(ftype)

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytearray] = []
scores: list[float] = []
toktypes: list[int] = []

tokenizer_json_file = dir_model / 'tokenizer.json'
if not tokenizer_json_file.is_file():
    print(f'Error: Missing {tokenizer_json_file}', file = sys.stderr)
    sys.exit(1)

# gpt2 tokenizer
gguf_writer.add_tokenizer_model("gpt2")

with open(tokenizer_json_file, "r", encoding="utf-8") as f:
    tokenizer_json = json.load(f)

print("gguf: get gpt2 tokenizer vocab")

# The number of tokens in tokenizer.json can differ from the expected vocab size.
# This causes downstream issues with mismatched tensor sizes when running the inference
vocab_size = hparams["vocab_size"] if "vocab_size" in hparams else len(tokenizer_json["model"]["vocab"])

# ref: https://github.com/cmp-nct/ggllm.cpp/blob/master/falcon_convert.py
tokenizer = AutoTokenizer.from_pretrained(dir_model)

reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

for i in range(vocab_size):
    if i in reverse_vocab:
        try:
            text = bytearray([byte_decoder[c] for c in reverse_vocab[i]])
        except KeyError:
            text = bytearray()
            for c in reverse_vocab[i]:
                if ord(c) < 256:  # single byte character
                    text.append(byte_decoder[ord(c)])
                else:  # multibyte special token character
                    text.extend(c.encode('utf-8'))
    else:
        print(f"Key {i} not in tokenizer vocabulary. Padding with an arbitrary token.")
        pad_token = f"[PAD{i}]".encode("utf8")
        text = bytearray(pad_token)

    tokens.append(text)
    scores.append(0.0)                      # dymmy
    toktypes.append(gguf.TokenType.NORMAL)  # dummy

gguf_writer.add_token_list(tokens)
gguf_writer.add_token_scores(scores)
gguf_writer.add_token_types(toktypes)

special_vocab = gguf.SpecialVocab(dir_model, load_merges = True)
special_vocab.add_to_gguf(gguf_writer)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# params for qkv transform
n_head    = hparams["n_head"]
n_head_kv = hparams["n_head_kv"] if "n_head_kv" in hparams else 1

head_dim = hparams["n_embd"] // n_head

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = iter(("pytorch_model.bin",))
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    model_part = torch.load(dir_model / part_name, map_location="cpu")

    for name in model_part.keys():
        data = model_part[name]

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
            print("Can not map tensor '" + name + "'")
            sys.exit()

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if ftype == 0 and data_dtype == np.float16:
            data = data.astype(np.float32)

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)

        print(name, "=>", new_name + ", shape = " + str(data.shape) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(new_name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
if not args.vocab_only:
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
