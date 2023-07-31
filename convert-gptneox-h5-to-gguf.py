# Quick and dirty HF gptneox--> gguf conversion

import gguf
import sys
import struct
import json
import numpy as np
from typing import Any, List
from pathlib import Path
from transformers import AutoModelForCausalLM


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)


# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"


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

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

if hparams["architectures"][0] != "GPTNeoXForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0] )
    sys.exit()

model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True, trust_remote_code=True)
list_vars = model.state_dict()

# count tensors to be converted
tensor_count = 0
for name in list_vars.keys():
    # we don't need these
    if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
        continue
    tensor_count += 1

gguf_writer = gguf.GGUFWriter.open(fname_out)

# This must be changed when adding/deleting kv
kv_count = 14

print("tensors " + str(tensor_count) + " kv " + str(kv_count))

print("write gguf header")

gguf_writer.write_header(tensor_count, kv_count)

print("write gguf hparams")

llm_arch = "gptneox"

gguf_writer.write_name("pythia-70b-deduped")
gguf_writer.write_description("gguf test model")
gguf_writer.write_architecture(llm_arch)
gguf_writer.write_context_length(llm_arch, hparams["max_position_embeddings"])
gguf_writer.write_embedding_length(llm_arch, hparams["hidden_size"])
gguf_writer.write_layer_count(llm_arch, hparams["num_hidden_layers"])
gguf_writer.write_feed_forward_length(llm_arch, hparams["intermediate_size"])
gguf_writer.write_rope_dimension_count(llm_arch, int( hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"])) )
gguf_writer.write_head_count(llm_arch, hparams["num_attention_heads"])
gguf_writer.write_parallel_residual(llm_arch, hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
gguf_writer.write_layer_norm_eps(llm_arch, hparams["layer_norm_eps"])

# TOKENIZATION

print("write gguf tokenizer")

tokens: List[str] = []
merges: List[str] = []

if Path(dir_model + "/tokenizer.json").is_file():
    # vocab type gpt2
    print("Adding gpt2 tokenizer vocab")

    with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    for key in tokenizer["model"]["vocab"]:
        tokens.append(key)

    merges = tokenizer["model"]["merges"]

gguf_writer.write_tokenizer_model("gpt2")
gguf_writer.write_token_list(tokens)
gguf_writer.write_token_merges(merges)

# TENSORS

# tensor info
print("write gguf tensor info")

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()

    # we don't need these
    if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
        continue

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if ftype != 0:
        if name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            ftype_cur = 0

    gguf_writer.write_tensor_info(name, data)


# tensor data
print("write gguf tensor data")

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Process tensor: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
        print("  Skip tensor: " + name)
        continue

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if ftype != 0:
        if name.endswith(".weight") and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    gguf_writer.write_tensor(data)

gguf_writer.close()


print("Done. Output file: " + fname_out)
print("")
