# Converts an RWKV model checkpoint to an rwkv.cpp compatible file.
# Usage: python convert_pytorch_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
# Get model checkpoints from https://huggingface.co/BlinkDL

# File format:
#
# RWKVModelFile {
#   // All ints and floats are in machine byte order.
#   // Magic is "ggml" string bytes.
#   int32 magic = 0x67676d66;
#   int32 version = 100;
#   int32 n_vocab;
#   int32 n_embed;
#   int32 n_layer;
#   // 0 if float32, 1 if float16.
#   int32 data_type;
#   // Read until EOF.
#   Parameter[] parameters;
# }
#
# Parameter {
#   int32 dim_count;
#   int32 key_length;
#   // 0 if float32, 1 if float16.
#   int32 data_type;
#   // Same values and order as in PyTorch's tensor.shape
#   int32[dim_count] shape;
#   // Keys are like "emb.weight", "block.0.ln1.weight".
#   uint8[key_length] key_utf8;
#   // Can be either float32 or float16.
#   float[product(shape)] data;
# }

import os
import argparse
import struct
import torch
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description='Convert an RWKV model checkpoint to an rwkv.cpp compatible file')
    parser.add_argument('src_path', help='Path to PyTorch checkpoint file')
    parser.add_argument('dest_path', help='Path to rwkv.cpp checkpoint file, will be overwritten')
    parser.add_argument('data_type', help='Data type, float16 or float32', type=str, choices=['float16', 'float32'], default='float32')
    return parser.parse_args()

def get_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
    n_layer = 0

    while f'blocks.{n_layer}.ln1.weight' in state_dict:
        n_layer += 1

    assert n_layer > 0

    return n_layer

def write_state_dict(state_dict: Dict[str, torch.Tensor], dest_path: str, data_type: str) -> None:
    emb_weight: torch.Tensor = state_dict['emb.weight']

    n_layer = get_layer_count(state_dict)
    n_vocab = emb_weight.shape[0]
    n_embed = emb_weight.shape[1]

    with open(dest_path, 'wb') as out_file:
        out_file.write(struct.pack(
            # Disable padding with '='
            '=iiiiii',
            # Magic: 'ggmf' in hex
            0x67676d66,
            # llama.cpp uses file versions 1+, let's use 100+ for rwkv.cpp
            100,
            n_vocab,
            n_embed,
            n_layer,
            1 if data_type == 'float16' else 0
        ))

        for k in state_dict.keys():
            tensor = state_dict[k].float()

            # Same processing as in "RWKV_in_150_lines.py"
            if '.time_' in k:
                # (1, 1, n_embed) -> (n_embed)
                tensor = tensor.squeeze()

            if '.time_decay' in k:
                tensor = -torch.exp(tensor)

            # Keep 1-dim vectors in fp32
            if data_type == 'float16' and len(tensor.shape) > 1:
                tensor = tensor.half()

            shape = tensor.shape

            print(f'Writing {k}, shape {shape}, type {tensor.dtype}')

            k_encoded: bytes = k.encode('utf-8')

            out_file.write(struct.pack(
                '=iii',
                len(shape),
                len(k_encoded),
                1 if tensor.dtype == torch.float16 else 0
            ))

            # Dimension order is reversed here:
            # * PyTorch shape is (x rows, y columns)
            # * ggml shape is (y elements in a row, x elements in a column)
            # Both shapes represent the same tensor.
            for dim in reversed(tensor.shape):
                out_file.write(struct.pack('=i', dim))

            out_file.write(k_encoded)

            tensor.numpy().tofile(out_file)

def main() -> None:
    args = parse_args()

    print(f'Reading {args.src_path}')

    state_dict: Dict[str, torch.Tensor] = torch.load(args.src_path, map_location='cpu')

    write_state_dict(state_dict, args.dest_path, args.data_type)

    print('Done')

# --- Tests ---

def test() -> None:
    test_file_path = 'convert_pytorch_rwkv_to_ggml_test.tmp'

    try:
        state_dict: Dict[str, torch.Tensor] = {
            'emb.weight': torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
            'blocks.0.ln1.weight': torch.tensor([1], dtype=torch.float32)
        }

        write_state_dict(state_dict, dest_path=test_file_path, data_type='float32')

        with open(test_file_path, 'rb') as input:
            actual_bytes: bytes = input.read()

        expected_bytes: bytes = struct.pack(
            '=iiiiii' + 'iiiii10sffffff' + 'iiii19sf',
            0x67676d66,
            100,
            3,
            2,
            1,
            0,
            # emb.weight
            2,
            10,
            0,
            2, 3,
            'emb.weight'.encode('utf-8'),
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            # blocks.0.ln1.weight
            1,
            19,
            0,
            1,
            'blocks.0.ln1.weight'.encode('utf-8'),
            1.0
        )

        assert list(actual_bytes) == list(expected_bytes), f'\nActual: {list(actual_bytes)}\nExpected: {list(expected_bytes)}'

        print('All tests pass')
    finally:
        if os.path.isfile(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    main()
