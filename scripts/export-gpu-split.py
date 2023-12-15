#!/usr/bin/env python3

import argparse
import torch
import torch.nn as tnn
from pathlib import Path
import os
import re
import struct
from typing import Any, BinaryIO
import numpy as np
import pickle

class ReluMLP(tnn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReluMLP, self).__init__()
        self.fc1 = tnn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = tnn.ReLU()
        self.fc2 = tnn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def _load_mlp_model(model_file: Path):
    model = torch.load(model_file)
    # hidden_size, input_size = model.get("fc1.weight").shape
    # output_size, _ = model.get("fc2.weight").shape
    # mlp = ReluMLP(input_size, hidden_size, output_size)
    # mlp.load_state_dict(model)
    return model


def load_mlp_predictors(models_base: Path):
    # TODO: might need a specification file to indicate which models to load.
    # But for now, let's assume it is a plain directory of models_{0, ... , n_layers - 1}.pt
    *_, files = next(os.walk(models_base))
    return [_load_mlp_model(models_base / f"activation_{i}.pt") for i in range(len(files))]


def write_file_header(fout: BinaryIO, n_tensors: int) -> None:
    fout.write(b"gglp"[::-1])  # magic (GGml mLP)
    fout.write(struct.pack("i", 1))  # file version
    # TODO: If we found we need more common parameters, we can add them here.
    fout.write(struct.pack("i", n_tensors))


def write_tensor_header(
    fout: BinaryIO, key: str, shape: tuple[int, ...], dtype: np.dtype
) -> None:
    _NUMPY_TYPE_TO_FTYPE: dict[str, int] = {"float32": 0, "float16": 1, "int32": 18}
    bkey = key.encode("utf-8")
    fout.write(
        struct.pack("iii", len(shape), len(bkey), _NUMPY_TYPE_TO_FTYPE[dtype.name])
    )
    fout.write(struct.pack("i" * len(shape), *shape))
    fout.write(bkey)
    # Aligns to 32 bytes
    fout.seek((fout.tell() + 31) & -32)


# TODO: need to add more details in key name to indicate the network, layer number, etc.
def _translate_mlp_key(key: str) -> str:
    match = re.match(r"^(fc\d+).weight$", key)
    if not match or len(match.groups()) != 1:
        raise ValueError(f"Unexpected key: {key}")
    return f"{match.group(1)}.weight.mlp"


def append_mlp_model(fout: BinaryIO, model: ReluMLP) -> None:
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        key = _translate_mlp_key(k)
        # torch.nn.Linear stores the weight matrix as (output_dim, input_dim), so does GGML.
        weights = v.half().detach().numpy()
        # GGML stores the weight matrix as (input_dim, output_dim)
        dims = weights.shape[::-1]
        print(
            f"{k} => {key} {weights.shape} {weights.dtype} {weights.nbytes/1024/1024} MiB"
        )
        # TODO: add option to write in float32
        write_tensor_header(fout, key, dims, np.dtype("float16"))
        weights.tofile(fout)

def append_gpu_idx(fout: BinaryIO, activation, select_count) -> None:
    values, indices = torch.topk(activation, k=int(select_count))
    gpu_idx = torch.zeros_like(activation)
    gpu_idx[indices] = 1
    gpu_idx = gpu_idx.numpy().astype(np.int32)
    weights = gpu_idx
    dims = gpu_idx.shape[::-1]
    key = "gpu_idx"
    print(
        f"{key} => {key} {weights.shape} {weights.dtype} {weights.nbytes/1024/1024} MiB"
    )
    write_tensor_header(fout, key, dims, np.dtype("int32"))
    weights.tofile(fout)

    indices = indices.numpy().astype(np.int32)
    weights = indices
    dims = weights.shape[::-1]
    key = "gpu_bucket"
    print(
        f"{key} => {key} {weights.shape} {weights.dtype} {weights.nbytes/1024/1024} MiB"
    )
    write_tensor_header(fout, key, dims, np.dtype("int32"))
    weights = np.sort(weights)
    weights.tofile(fout)

def main(predictors_path: str, output_path: str, solver_path: str):
    predictors = load_mlp_predictors(Path(predictors_path)) # predictor => activation acount
    n_tensors = len(predictors) * 2 # gpu_idx and gpu_bucket
    print(f"found {len(predictors)} MLP adapters with {n_tensors} tensors")
    with open(solver_path, "rb") as f:
        loaded_lst = pickle.load(f)
        # print(f"check solver {loaded_lst}")
    with open(output_path, "wb") as fout:
        fout.truncate()
        write_file_header(fout, n_tensors=n_tensors)
        for i, activation in enumerate(predictors):
            print(f"appending gpu idx layer-{i}")
            append_gpu_idx(fout, activation, loaded_lst[i])
            # append_gpu_idx(fout, activation, (32768*0.0))

    print(f"converted MLP adapters from {predictors_path} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictors_path", help="path to the MLP predictors")
    parser.add_argument(
        "output_path",
        help="path to the output GGML adapter",
        default="./gpu-index.bin",
    )
    parser.add_argument("solver", help="path to the solver")

    args = parser.parse_args()
    main(args.predictors_path, args.output_path, args.solver)
