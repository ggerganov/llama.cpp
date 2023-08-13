# HF llama --> gguf conversion, GQA/70b not supported

import gguf
import gguf_tensor_map as tmap
import os
import sys
import struct
import json
import numpy as np
from typing import Any, List
from pathlib import Path
import torch
from sentencepiece import SentencePieceProcessor


#NDArray = np.ndarray[Any, Any]
# compatible with python < 3.9
NDArray: 'TypeAlias' = 'np.ndarray[Any, Any]'


def permute(weights: NDArray, n_head: int) -> NDArray:
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
#
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

gguf_writer = gguf.GGUFWriter.open(fname_out)


print("gguf: get model metadata")

llm_arch = "llama"
hf_repo = hparams["_name_or_path"]
head_count = hparams["num_attention_heads"]
head_count_kv = hparams["num_key_value_heads"]
block_count = hparams["num_hidden_layers"]

gguf_writer.add_name(last_dir)
gguf_writer.add_architecture(llm_arch)
guff_writer.add_source_hf_repo(hf_repo)
gguf_writer.add_context_length(llm_arch, hparams["max_position_embeddings"])
gguf_writer.add_embedding_length(llm_arch, hparams["hidden_size"])
gguf_writer.add_block_count(llm_arch, block_count)
gguf_writer.add_feed_forward_length(llm_arch, hparams["intermediate_size"])
gguf_writer.add_rope_dimension_count(llm_arch, hparams["hidden_size"] // hparams["num_attention_heads"])
gguf_writer.add_head_count(llm_arch, head_count)
gguf_writer.add_head_count_kv(llm_arch, head_count_kv)
gguf_writer.add_layer_norm_rms_eps(llm_arch, hparams["rms_norm_eps"])


# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: List[str] = []
scores: List[float] = []

if Path(dir_model + "/tokenizer.model").is_file():
    # vocab type sentencepiece
    print("gguf: get sentencepiece tokenizer vocab and scores")

    tokenizer = SentencePieceProcessor(dir_model + "/tokenizer.model")

    for i in range(tokenizer.vocab_size()):
        text: bytes
        if tokenizer.is_unknown(i):
            text = " \u2047 ".encode("utf-8")
        elif tokenizer.is_control(i):
            text = b""
        if tokenizer.is_byte(i):
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                raise Exception(f"Invalid token: {piece}")
            byte_value = int(piece[3:-1], 16)
            text = struct.pack("B", byte_value)
        else:
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        score: float = tokenizer.get_score(i)

        tokens.append(text)
        scores.append(score)

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)

if Path(dir_model + "/tokenizer.json").is_file():
    with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    if "added_tokens" in tokenizer and Path(dir_model + "/tokenizer_config.json").is_file():
        print("gguf: get special token ids")

        with open(dir_model + "/tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        # find special token ids

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


# TENSORS

tensor_map = tmap.get_tensor_map(block_count)

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

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

        # permute these
        if name.endswith(".q_proj.weight") or name.endswith(".k_proj.weight"):
            data = permute(data, head_count)

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
        if ftype == 0 and data.dtype == np.float16:
            data_dtype = np.float32

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data_dtype = np.float32

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if ftype == 1 and data.dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data_dtype = np.float16

        data_nbytes = data.size * 2 if data_dtype == np.float16 else data.size * 4

        gguf_writer.add_tensor_info(name, data.shape, data_dtype, data_nbytes)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
print("gguf: write tensor metadata")
gguf_writer.write_ti_data_to_file()

# tensor data
print("gguf: convert and write tensor data")

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

<< << << < HEAD
   n_dims = len(data.shape)
    data_dtype = data.dtype
== == == =
   old_dtype = data.dtype

    # we don't need these
    if name.endswith(".rotary_emb.inv_freq"):
        continue
>>>>>> > 17800cd80fec468411481dc34a51d42a936442f1

   # convert any unsupported data types to float32
   if data.dtype != torch.float16 and data.dtype != torch.float32:
        data = data.to(torch.float32)

    data = data.squeeze().numpy()

    # permute these
    if name.endswith(".q_proj.weight") or name.endswith(".k_proj.weight"):
        data = permute(data, head_count)

    # map tensor names
    if name.endswith(".weight") and name[:-7] in tensor_map:
        name = tensor_map[name[:-7]] + ".weight"
    elif name.endswith(".bias") and name[:-5] in tensor_map:
        name = tensor_map[name[:-5]] + ".bias"
    else:
        print("Can not map tensor '" + name + "'" )
        sys.exit()

    n_dims = len(data.shape)
    data_dtype = data.dtype

    # if f32 desired, convert any float16 to float32
    if ftype == 0 and data.dtype == np.float16:
        data = data.astype(np.float32)

    # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
    if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
        data = data.astype(np.float32)

    # if f16 desired, convert any float32 2-dim weight tensors to float16
    if ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
        data = data.astype(np.float16)

    print(name + ", shape " + str(len(data.shape)) + ", " + str(old_dtype) + " --> " + str(data.dtype))

    gguf_writer.write_tensor_to_file(data)

gguf_writer.close()


print("gguf: model successfully exported to '" + fname_out + "'")
print("")
