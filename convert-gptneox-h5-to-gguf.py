# Quick and dirty HF gptneox--> gguf conversion

import gguf
import gguf_tensor_map as tmap
import os
import sys
import struct
import json
import numpy as np
from typing import Any, List
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
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

if hparams["architectures"][0] != "GPTNeoXForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0] )
    sys.exit()


model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True, trust_remote_code=True)
list_vars = model.state_dict()

gguf_writer = gguf.GGUFWriter.open(fname_out)

print("gguf: get model metadata")

llm_arch    = "gptneox"
block_count = hparams["num_hidden_layers"]

gguf_writer.add_name(last_dir)
gguf_writer.add_architecture(llm_arch)
gguf_writer.add_context_length(llm_arch, hparams["max_position_embeddings"])
gguf_writer.add_embedding_length(llm_arch, hparams["hidden_size"])
gguf_writer.add_layer_count(llm_arch, block_count)
gguf_writer.add_feed_forward_length(llm_arch, hparams["intermediate_size"])
gguf_writer.add_rope_dimension_count(llm_arch, int( hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"])) )
gguf_writer.add_head_count(llm_arch, hparams["num_attention_heads"])
gguf_writer.add_parallel_residual(llm_arch, hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
gguf_writer.add_layer_norm_eps(llm_arch, hparams["layer_norm_eps"])

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: List[str] = []
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

    vocab_size = len( tokenizer_json["model"]["vocab"] )

    # from ggllm.cpp falcon_convert.py
    tokenizer = AutoTokenizer.from_pretrained(dir_model)

    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v:k for k, v in byte_encoder.items()}

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
            padding_token = f"[PAD{i}]".encode("utf8")
            text = bytearray(padding_token)
        tokens.append(text)

    gguf_writer.add_token_list(tokens)

    if "added_tokens" in tokenizer_json and Path(dir_model + "/tokenizer_config.json").is_file():
        print("gguf: get special token ids")

        with open(dir_model + "/tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        # find special token ids

        if "bos_token" in tokenizer_config:
            for key in tokenizer_json["added_tokens"]:
                if key["content"] == tokenizer_config["bos_token"]:
                    gguf_writer.add_bos_token_id(key["id"])

        if "eos_token" in tokenizer_config:
            for key in tokenizer_json["added_tokens"]:
                if key["content"] == tokenizer_config["eos_token"]:
                    gguf_writer.add_eos_token_id(key["id"])

        if "unk_token" in tokenizer_config:
            for key in tokenizer_json["added_tokens"]:
                if key["content"] == tokenizer_config["unk_token"]:
                    gguf_writer.add_unk_token_id(key["id"])

        if "sep_token" in tokenizer_config:
            for key in tokenizer_json["added_tokens"]:
                if key["content"] == tokenizer_config["sep_token"]:
                    gguf_writer.add_sep_token_id(key["id"])

        if "pad_token" in tokenizer_config:
            for key in tokenizer_json["added_tokens"]:
                if key["content"] == tokenizer_config["pad_token"]:
                    gguf_writer.add_pad_token_id(key["id"])


# TENSORS

tensor_map = tmap.get_tensor_map(block_count)

# tensor info
print("gguf: get tensor metadata")

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()

    # we don't need these
    if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
        continue

    # map tensor names
    if name.endswith(".weight") and name[:-7] in tensor_map:
        name = tensor_map[name[:-7]] + ".weight"
    elif name.endswith(".bias") and name[:-5] in tensor_map:
        name = tensor_map[name[:-5]] + ".bias"
    else:
        print( "Can not map tensor '" + name + "'" )
        sys.exit()

    n_dims = len(data.shape)
    data_dtype = data.dtype 

#    print( name + " dims " + str(n_dims) + " dtype " + str(data.dtype) )

    if data.dtype != np.float16 and data.dtype != np.float32:
        # convert any unsupported data types to float32
        data_dtype = np.float32
    elif ftype == 1 and data.dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
        # if f16 desired, convert any float32 2-dim weight tensors to float16
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

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()

    # we don't need these
    if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
        continue

    n_dims = len(data.shape)
    data_dtype = data.dtype 

    if data_dtype != np.float16 and data_dtype != np.float32:
        # convert any unsupported data types to float32
        data = data.astype(np.float32)
    elif ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
        # if f16 desired, convert any float32 2-dim weight tensors to float16
        data = data.astype(np.float16)

    gguf_writer.write_tensor_to_file(data)

gguf_writer.close()


print("gguf: model successfully exported to '" + fname_out + "'" )
print("")
