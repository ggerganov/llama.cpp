#!/usr/bin/env python3
# HF falcon--> gguf conversion

import gguf
import os
import sys
import struct
import json
import numpy as np
import torch

from typing import Any, List
from pathlib import Path
from transformers import AutoTokenizer

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
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def count_model_parts(dir_model: str) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)


# output in the same directory as the model
dir_model = sys.argv[1]
last_dir = os.path.basename(os.path.normpath(dir_model))

# possible tensor data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16

# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))

        sys.exit(1)

fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".gguf"

print("gguf: loading model "+last_dir)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

if hparams["architectures"][0] != "RWForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.FALCON
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["n_layer"]

gguf_writer.add_name("Falcon")
gguf_writer.add_context_length(2048) # not in config.json
gguf_writer.add_tensor_data_layout("jploski") # qkv tensor transform
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_feed_forward_length(4 * hparams["hidden_size"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_head_count(hparams["n_head"])
if "n_head_kv" in hparams:
    gguf_writer.add_head_count_kv(hparams["n_head_kv"])
else:
    gguf_writer.add_head_count_kv(1)
gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])
gguf_writer.add_file_type(ftype)

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: List[str] = []
scores: List[float] = []
toktypes: List[int] = []
merges: List[str] = []


if Path(dir_model + "/tokenizer.json").is_file():
    # gpt2 tokenizer
    gguf_writer.add_tokenizer_model("gpt2")

    print("gguf: get gpt2 tokenizer merges")

    with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    merges = tokenizer_json["model"]["merges"]

    gguf_writer.add_token_merges(merges)

    print("gguf: get gpt2 tokenizer vocab")

    vocab_size = len(tokenizer_json["model"]["vocab"])

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

print("gguf: get special token ids")
# Look for special tokens in config.json

if "bos_token_id" in hparams and hparams["bos_token_id"] != None:
    gguf_writer.add_bos_token_id(hparams["bos_token_id"])

if "eos_token_id" in hparams and hparams["eos_token_id"] != None:
    gguf_writer.add_eos_token_id(hparams["eos_token_id"])

if "unk_token_id" in hparams and hparams["unk_token_id"] != None:
    gguf_writer.add_unk_token_id(hparams["unk_token_id"])

if "sep_token_id" in hparams and hparams["sep_token_id"] != None:
    gguf_writer.add_sep_token_id(hparams["sep_token_id"])

if "pad_token_id" in hparams and hparams["pad_token_id"] != None:
    gguf_writer.add_pad_token_id(hparams["pad_token_id"])


# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# params for qkv transform
n_head    = hparams["n_head"]
n_head_kv = hparams["n_head_kv"] if "n_head_kv" in hparams else 1

head_dim = hparams["hidden_size"] // n_head

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = ("pytorch_model.bin",)
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    print("gguf: loading model part '" + part_name + "'")
    model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")

    for name in model_part.keys():
        data = model_part[name]

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        # QKV tensor transform
        # The original query_key_value tensor contains n_head_kv "kv groups",
        # each consisting of n_head/n_head_kv query weights followed by one key
        # and one value weight (shared by all query heads in the kv group).
        # This layout makes it a big pain to work with in GGML.
        # So we rearrange them here,, so that we have n_head query weights
        # followed by n_head_kv key weights followed by n_head_kv value weights,
        # in contiguous fashion.
        # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

        if "query_key_value" in name:
            qkv = data.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
            q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
            data = torch.cat((q,k,v)).reshape_as(data)

        data = data.squeeze().numpy()

        # map tensor names
        if name.endswith(".weight") and name[:-7] in tensor_map:
            name = tensor_map[name[:-7]] + ".weight"
        elif name.endswith(".bias") and name[:-5] in tensor_map:
            name = tensor_map[name[:-5]] + ".bias"
        else:
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

        print(name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
print("gguf: write tensors")
gguf_writer.write_tensors_to_file()

gguf_writer.close()

print("gguf: model successfully exported to '" + fname_out + "'")
print("")
