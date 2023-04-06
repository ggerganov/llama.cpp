import os
import re
import struct
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


# TODO: import this from convert.py once #545 is merged
@dataclass(frozen=True)
class UnquantizedDataType:
    name: str

DT_F16 = UnquantizedDataType('F16')
DT_F32 = UnquantizedDataType('F32')

@dataclass(frozen=True)
class QuantizedDataType:
    groupsize: int
    have_addends: bool
    have_g_idx: bool

DataType = UnquantizedDataType

DATA_TYPE_TO_FTYPE: dict[DataType, int] = {
    DT_F32: 0,
    DT_F16: 1,
}

DATA_TYPE_TO_NUMPY: dict[DataType, np.dtype[Any]] = {
    DT_F16: np.dtype(np.float16),
    DT_F32: np.dtype(np.float32),
}

NUMPY_TYPE_TO_DATA_TYPE: dict[np.dtype[Any], DataType] = {dtype: data_type for (data_type, dtype) in DATA_TYPE_TO_NUMPY.items()}

HF_SUBLAYER_TO_GGML = {
    "self_attn.q_proj": "attention.wq.weight",
    "self_attn.k_proj": "attention.wk.weight",
    "self_attn.v_proj": "attention.wv.weight",
    "self_attn.o_proj": "attention.wo.weight",
}

def translate_tensor_name(t):
    match = re.match(r'.*layers\.(\d+)\.(\w+\.\w+)\.lora_(A|B)\.weight', t)
    if match:
        nn = match.group(1)
        sub_layer = match.group(2)
        lora_type = match.group(3)

        sub_layer_renamed = HF_SUBLAYER_TO_GGML.get(sub_layer)
        if sub_layer_renamed is None:
            print(f"Error: unrecognized sub-layer {sub_layer} in tensor {t}")
            exit(1)

        output_string = f"layers.{nn}.{HF_SUBLAYER_TO_GGML[sub_layer]}.lora{lora_type}"
        return output_string
    else:
        print(f"Error: unrecognized tensor {t}")
        exit(1)

def write_file_header(fout):
    fout.write(b"ggla"[::-1]) # magic (ggml lora)
    fout.write(struct.pack("i", 1)) # file version


def write_tensor_header(self, name: str, shape: Sequence[int], data_type: 1) -> None:
    sname = name.encode('utf-8')
    fout.write(struct.pack("iii", len(shape), len(sname), DATA_TYPE_TO_FTYPE[NUMPY_TYPE_TO_DATA_TYPE[data_type]]))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)
    

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} adapter_model.bin [ggml_adapter_model.bin]")
    sys.exit(1)

input_path = sys.argv[1]
if len(sys.argv) > 2:
    output_path = sys.argv[2]
else:
    output_filename = f"ggml_{os.path.basename(input_path)}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

model = torch.load(input_path, map_location="cpu")

with open(output_path, "wb") as fout:
    write_file_header(fout)
    for k, v in model.items():
        # since ggml doesn't always support other types for the second operand,
        # the tensors are always converted and exported as f32
        t = v.float().numpy()
        print(f"{k} => {translate_tensor_name(k)} {t.shape} {t.dtype} {t.nbytes/1024/1024:.2f}MB")
        write_tensor_header(fout, translate_tensor_name(k), t.shape, t.dtype)
        t.tofile(fout)

print(f"Converted {input_path} to {output_path}")