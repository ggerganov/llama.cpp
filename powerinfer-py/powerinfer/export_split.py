import argparse
import pickle
import gguf
from gguf.constants import GGMLQuantizationType
from gguf.gguf_writer import GGUFWriter
import torch
from pathlib import Path
import os
import struct
import numpy as np

def load_activation_weights(models_base: Path):
    # TODO: might need a specification file to indicate which models to load.
    # But for now, let's assume it is a plain directory of activation_{0, ... , n_layers - 1}.pt
    *_, files = next(os.walk(models_base))
    return [torch.load(models_base / f"activation_{i}.pt") for i in range(len(files))]

def append_gpu_idx(gguf: GGUFWriter, i_layer: int, activation, select_count) -> None:
    _, indices = torch.topk(activation, k=int(select_count))
    gpu_idx = torch.zeros_like(activation)
    gpu_idx[indices] = 1
    gpu_idx = gpu_idx.numpy().astype(np.int32)
    key = f"blk.{i_layer}.gpu_idx"
    print(
        f"{key} => {key} {gpu_idx.shape} {gpu_idx.dtype} {gpu_idx.nbytes/1024/1024} MiB"
    )
    gguf.add_tensor(
        name=key,
        tensor=gpu_idx,
        raw_shape=gpu_idx.shape[::-1],
        raw_dtype=GGMLQuantizationType.I32,
    )

    indices = indices.numpy().astype(np.int32)
    gpu_bucket = np.sort(indices)
    key = f"blk.{i_layer}.gpu_bucket"
    print(
        f"{key} => {key} {gpu_bucket.shape} {gpu_bucket.dtype} {gpu_bucket.nbytes/1024/1024} MiB"
    )
    gguf.add_tensor(
        name=key,
        tensor=gpu_bucket,
        raw_shape=gpu_bucket.shape[::-1],
        raw_dtype=GGMLQuantizationType.I32,
    )

def export_split(activations_path: str, output_path: str, solved_list: list[int], vram_capacity: int):
    predictors = load_activation_weights(Path(activations_path)) # predictor => activation acount
    gguf_out = GGUFWriter(output_path, "generic.gpu_index")
    for i, (activation, selected_count) in enumerate(zip(predictors, solved_list)):
        append_gpu_idx(gguf_out, i, activation, selected_count)

    # set kvs
    gguf_out.add_block_count(len(predictors))
    # TODO: better to save the actual capacity that split neurons require
    gguf_out.add_uint64(gguf.Keys.Split.VRAM_CAPACITY, vram_capacity)

    gguf_out.write_header_to_file()
    gguf_out.write_kv_data_to_file()
    gguf_out.write_tensors_to_file()
    gguf_out.close()

    # post-process: write another unique file header to distinguish from the origianl GGUF file
    with open(output_path, "r+b") as fout:
        POWERINFER_MAGIC = int.from_bytes(b"PWRI", "little")
        fout.write(struct.pack("<I", POWERINFER_MAGIC))
        fout.write(struct.pack("<I", 3))

    print(f"exported GPU index to {output_path}")

