# Quick and dirty HF llama --> gguf conversion, GQA/70b wont work

import gguf
import sys
import struct
import json
import numpy as np
from typing import Any, List
from pathlib import Path
from transformers import AutoModelForCausalLM
from sentencepiece import SentencePieceProcessor


NDArray = np.ndarray[Any, Any]


def permute(weights: NDArray, n_head: int) -> NDArray:
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))


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

if hparams["architectures"][0] != "LlamaForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0] )
    sys.exit()

model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True, trust_remote_code=True)
list_vars = model.state_dict()

# count tensors to be converted
tensor_count = 0
for name in list_vars.keys():
    # we don't need these
    if name.endswith(".rotary_emb.inv_freq"):
        continue
    tensor_count += 1

gguf_writer = gguf.GGUFWriter.open(fname_out)

# This must be changed when adding/deleting kv
kv_count = 13

print("tensors " + str(tensor_count) + " kv " + str(kv_count))

print("write gguf header")

gguf_writer.write_header(tensor_count, kv_count)

print("write gguf hparams")

llm_arch = "llama"

gguf_writer.write_name("llama2-7b")
gguf_writer.write_description("gguf test model")
gguf_writer.write_architecture(llm_arch)
gguf_writer.write_context_length(llm_arch, hparams["max_position_embeddings"])
gguf_writer.write_embedding_length(llm_arch, hparams["hidden_size"])
gguf_writer.write_layer_count(llm_arch, hparams["num_hidden_layers"])
gguf_writer.write_feed_forward_length(llm_arch, hparams["intermediate_size"])
gguf_writer.write_rope_dimension_count(llm_arch, hparams["hidden_size"] // hparams["num_attention_heads"])
gguf_writer.write_head_count(llm_arch, hparams["num_attention_heads"])
gguf_writer.write_layer_norm_rms_eps(llm_arch, hparams["rms_norm_eps"])


# TOKENIZATION

print("write gguf tokenizer")

tokens: List[str] = []
scores: List[float] = []

if Path(dir_model + "/tokenizer.model").is_file():
    # vocab type sentencepiece
    print("Adding sentencepiece tokenizer vocab.")
    tokenizer = SentencePieceProcessor(dir_model + "/tokenizer.model")

    # output vocab_size followed by all piece/score pairs
    outbytes: bytes
    outbytes = b""
    outbytes += struct.pack("I", tokenizer.vocab_size())

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

gguf_writer.write_tokenizer_model("llama")
gguf_writer.write_token_list(tokens)
gguf_writer.write_token_scores(scores)

# TENSORS

# tensor info
print("write gguf tensor info")

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()

    # we don't need these
    if name.endswith(".rotary_emb.inv_freq"):
        continue

    # permute these
    if name.endswith(".q_proj.weight") or name.endswith(".k_proj.weight"):
        data = permute(data, hparams["num_attention_heads"])

    # chnage tensor name

    if name == "model.embed_tokens.weight":
        name = "tok_embeddings.weight"
    elif name == "model.norm.weight":
        name = "norm.weight"
    elif name == "lm_head.weight":
        name = "output.weight"
    else:
        for i in range(80):  # maximum number of layers
            if name == "model.layers." + str(i) + ".input_layernorm.weight":
                name = "layers." + str(i) + ".attention_norm.weight"
                break
            if name == "model.layers." + str(i) + ".self_attn.q_proj.weight":
                name = "layers." + str(i) + ".attention.wq.weight"
                break
            if name == "model.layers." + str(i) + ".self_attn.k_proj.weight":
                name = "layers." + str(i) + ".attention.wk.weight"
                break
            if name == "model.layers." + str(i) + ".self_attn.v_proj.weight":
                name = "layers." + str(i) + ".attention.wv.weight"
                break
            if name == "model.layers." + str(i) + ".self_attn.o_proj.weight":
                name = "layers." + str(i) + ".attention.wo.weight"
                break
            if name == "model.layers." + str(i) + ".post_attention_layernorm.weight":
                name = "layers." + str(i) + ".ffn_norm.weight"
                break
            if name == "model.layers." + str(i) + ".mlp.gate_proj.weight":
                name = "layers." + str(i) + ".feed_forward.w1.weight"
                break
            if name == "model.layers." + str(i) + ".mlp.down_proj.weight":
                name = "layers." + str(i) + ".feed_forward.w2.weight"
                break
            if name == "model.layers." + str(i) + ".mlp.up_proj.weight":
                name = "layers." + str(i) + ".feed_forward.w3.weight"
                break

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
    if name.endswith(".rotary_emb.inv_freq"):
        print("  Skip tensor: " + name)
        continue

    # permute these
    if name.endswith(".q_proj.weight") or name.endswith(".k_proj.weight"):
        print("  Permute tensor: " + name)
        data = permute(data, hparams["num_attention_heads"])

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
