import json
import os
import re
import struct
import sys
from typing import Any, Dict, Sequence, TextIO

import torch

from convert import DATA_TYPE_TO_FTYPE, NUMPY_TYPE_TO_DATA_TYPE, DataType

HF_SUBLAYER_TO_GGML = {
    "self_attn.q_proj": "attention.wq",
    "self_attn.k_proj": "attention.wk",
    "self_attn.v_proj": "attention.wv",
    "self_attn.o_proj": "attention.wo",
    "mlp.gate_proj": "feed_forward.w1",
    "mlp.down_proj": "feed_forward.w2",
    "mlp.up_proj": "feed_forward.w3",
    "input_layernorm": "attention_norm",
    "post_attention_layernorm": "ffn_norm",
    # "norm": "norm",
    # "embed_tokens": "tok_embeddings",
    # "lm_head": "output",
}


def translate_tensor_name(t: str) -> str:
    match = re.match(r".*layers\.(\d+)\.(\w+\.\w+)\.lora_(A|B)\.weight", t)
    if match:
        nn = match.group(1)
        sub_layer = match.group(2)
        lora_type = match.group(3)

        sub_layer_renamed = HF_SUBLAYER_TO_GGML.get(sub_layer)
        if sub_layer_renamed is None:
            print(f"Error: unrecognized sub-layer {sub_layer} in tensor {t}")
            sys.exit(1)

        output_string = (
            f"layers.{nn}.{HF_SUBLAYER_TO_GGML[sub_layer]}.weight.lora{lora_type}"
        )
        return output_string
    else:
        print(f"Error: unrecognized tensor {t}")
        sys.exit(1)


def write_file_header(fout: TextIO, params: Dict[str, Any]) -> None:
    fout.write(b"ggla"[::-1])  # magic (ggml lora)
    fout.write(struct.pack("i", 1))  # file version
    fout.write(struct.pack("i", params["r"]))
    # https://opendelta.readthedocs.io/en/latest/modules/deltas.html says that `lora_alpha` is an int
    # but some models ship a float value instead
    # let's convert to int, but fail if lossless conversion is not possible
    assert int(params["lora_alpha"]) == params["lora_alpha"], "cannot convert float to int losslessly"
    fout.write(struct.pack("i", int(params["lora_alpha"])))


def write_tensor_header(
    self, name: str, shape: Sequence[int], data_type: DataType
) -> None:
    sname = name.encode("utf-8")
    fout.write(
        struct.pack(
            "iii",
            len(shape),
            len(sname),
            DATA_TYPE_TO_FTYPE[NUMPY_TYPE_TO_DATA_TYPE[data_type]],
        )
    )
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <path>")
    print(
        "Path must contain HuggingFace PEFT LoRA files 'adapter_config.json' and 'adapter_model.bin'"
    )
    sys.exit(1)

input_json = os.path.join(sys.argv[1], "adapter_config.json")
input_model = os.path.join(sys.argv[1], "adapter_model.bin")
output_path = os.path.join(sys.argv[1], "ggml-adapter-model.bin")

model = torch.load(input_model, map_location="cpu")

with open(input_json, "r") as f:
    params = json.load(f)

if params["peft_type"] != "LORA":
    print(f"Error: unsupported adapter type {params['peft_type']}, expected LORA")
    sys.exit(1)

if params["fan_in_fan_out"] is True:
    print("Error: param fan_in_fan_out is not supported")
    sys.exit(1)

if params["bias"] is not None and params["bias"] != "none":
    print("Error: param bias is not supported")
    sys.exit(1)

# TODO: these seem to be layers that have been trained but without lora.
# doesn't seem widely used but eventually should be supported
if params["modules_to_save"] is not None and len(params["modules_to_save"]) > 0:
    print("Error: param modules_to_save is not supported")
    sys.exit(1)

with open(output_path, "wb") as fout:
    fout.truncate()

    write_file_header(fout, params)
    for k, v in model.items():
        if k.endswith("lora_A.weight"):
            if v.dtype != torch.float16 and v.dtype != torch.float32:
                v = v.float()
            v = v.T
        else:
            v = v.float()

        t = v.numpy()
        tname = translate_tensor_name(k)
        print(f"{k} => {tname} {t.shape} {t.dtype} {t.nbytes/1024/1024:.2f}MB")
        write_tensor_header(fout, tname, t.shape, t.dtype)
        t.tofile(fout)

print(f"Converted {input_json} and {input_model} to {output_path}")
