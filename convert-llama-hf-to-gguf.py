#!/usr/bin/env python3
# HF llama --> gguf conversion

import gguf
import os
import sys
import struct
import json
import numpy as np
import torch

from typing import Any, List, Optional
from pathlib import Path
from sentencepiece import SentencePieceProcessor

#NDArray = np.ndarray[Any, Any]
# compatible with python < 3.9
NDArray: 'TypeAlias' = 'np.ndarray[Any, Any]'

# reverse HF permute back to original pth layout
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py


def reverse_hf_permute(weights: NDArray, n_head: int, n_kv_head: Optional[int] = None) -> NDArray:
    if n_kv_head is not None and n_head != n_kv_head:
        n_head //= n_kv_head

    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


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

if hparams["architectures"][0] != "LlamaForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.LLAMA
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_hidden_layers"]
head_count = hparams["num_attention_heads"]

if "num_key_value_heads" in hparams:
    head_count_kv = hparams["num_key_value_heads"]
else:
    head_count_kv = head_count

if "_name_or_path" in hparams:
    hf_repo = hparams["_name_or_path"]
else:
    hf_repo = ""

if "max_sequence_length" in hparams:
    ctx_length = hparams["max_sequence_length"]
elif "max_position_embeddings" in hparams:
    ctx_length = hparams["max_position_embeddings"]
else:
    print("gguf: can not find ctx length parameter.")

    sys.exit()


gguf_writer.add_name(last_dir)
gguf_writer.add_source_hf_repo(hf_repo)
gguf_writer.add_tensor_data_layout("Meta AI original pth")
gguf_writer.add_context_length(ctx_length)
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])
gguf_writer.add_head_count(head_count)
gguf_writer.add_head_count_kv(head_count_kv)
gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

if "rope_scaling" in hparams and hparams["rope_scaling"] != None and "factor" in hparams["rope_scaling"]:
    if "type" in hparams["rope_scaling"]:
        if hparams["rope_scaling"]["type"] == "linear":
            gguf_writer.add_rope_scale_linear(hparams["rope_scaling"]["factor"])


# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: List[bytes] = []
scores: List[float] = []
toktypes: List[int] = []

if Path(dir_model + "/tokenizer.model").is_file():
    # vocab type sentencepiece
    print("gguf: get sentencepiece tokenizer vocab, scores and token types")

    tokenizer = SentencePieceProcessor(dir_model + "/tokenizer.model")

    for i in range(tokenizer.vocab_size()):
        text: bytes
        score: float

        piece = tokenizer.id_to_piece(i)
        text = piece.encode("utf-8")
        score = tokenizer.get_score(i)

        toktype = 1  # defualt to normal token type
        if tokenizer.is_unknown(i):
            toktype = 2
        if tokenizer.is_control(i):
            toktype = 3

        # toktype = 4 is user-defined = tokens from added_tokens.json

        if tokenizer.is_unused(i):
            toktype = 5
        if tokenizer.is_byte(i):
            toktype = 6

        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    if Path(dir_model + "/added_tokens.json").is_file():
        with open(dir_model + "/added_tokens.json", "r", encoding="utf-8") as f:
            addtokens_json = json.load(f)

            print("gguf: get added tokens")

            for key in addtokens_json:
                tokens.append( key.encode("utf-8") )
                scores.append(-1000.0)
                toktypes.append(4) # user-defined token type


    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)


print("gguf: get special token ids")

if Path(dir_model + "/tokenizer.json").is_file():
    # Look for special tokens in tokenizer.json if it exists

    with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    if "added_tokens" in tokenizer and Path(dir_model + "/tokenizer_config.json").is_file():

        with open(dir_model + "/tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        if "bos_token" in tokenizer_config and tokenizer_config["bos_token"] != None:
            for key in tokenizer["added_tokens"]:
                if key["content"] == tokenizer_config["bos_token"]["content"]:
                    gguf_writer.add_bos_token_id(key["id"])

        if "eos_token" in tokenizer_config and tokenizer_config["eos_token"] != None:
            for key in tokenizer["added_tokens"]:
                if key["content"] == tokenizer_config["eos_token"]["content"]:
                    gguf_writer.add_eos_token_id(key["id"])

        if "unk_token" in tokenizer_config and tokenizer_config["unk_token"] != None:
            for key in tokenizer["added_tokens"]:
                if key["content"] == tokenizer_config["unk_token"]["content"]:
                    gguf_writer.add_unk_token_id(key["id"])

        if "sep_token" in tokenizer_config and tokenizer_config["sep_token"] != None:
            for key in tokenizer["added_tokens"]:
                if key["content"] == tokenizer_config["sep_token"]["content"]:
                    gguf_writer.add_sep_token_id(key["id"])

        if "pad_token" in tokenizer_config and tokenizer_config["pad_token"] != None:
            for key in tokenizer["added_tokens"]:
                if key["content"] == tokenizer_config["pad_token"]["content"]:
                    gguf_writer.add_pad_token_id(key["id"])
else:
    # If no tokenizer.json: Look for special tokens in config.json

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

        # we don't need these
        if name.endswith(".rotary_emb.inv_freq"):
            continue

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

        # reverse permute these
        if name.endswith(".q_proj.weight"):
            data = reverse_hf_permute(data, head_count)
        if name.endswith(".k_proj.weight"):
            data = reverse_hf_permute(data, head_count, head_count_kv)

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
