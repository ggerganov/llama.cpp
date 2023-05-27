# Converts an RWKV model checkpoint in PyTorch format to an rwkv.cpp compatible file.
# Usage: python convert_pytorch_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
# Get model checkpoints from https://huggingface.co/BlinkDL
# See FILE_FORMAT.md for the documentation on the file format.

import argparse
import struct
import torch
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description='Convert an RWKV model checkpoint in PyTorch format to an rwkv.cpp compatible file')
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
            101,
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

if __name__ == "__main__":
    main()