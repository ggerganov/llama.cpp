#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import os

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf


def convert_byteorder(filename: str, order: str) -> None:
    if order not in ("big", "little", "native"):
        raise ValueError(f"Bad order parameter {order}")
    reader = gguf.GGUFReader(filename, "r+")
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # Host is little endian
        host_endian = "little"
        swapped_endian = "big"
    else:
        # Sorry PDP or other weird systems that don't use BE or LE.
        host_endian = "big"
        swapped_endian = "little"
    if reader.byte_order == "S":
        file_endian = swapped_endian
    else:
        file_endian = host_endian
    if order == "native":
        order = host_endian
    print(
        f"* Host is {host_endian.upper()} endian, GGUF file seems to be {file_endian.upper()} endian"
    )
    if file_endian == order:
        print(f"* File is already {order.upper()} endian. Nothing to do.")
        sys.exit(0)
    print("* Checking tensors for conversion compatibility")
    for tensor in reader.tensors:
        if tensor.tensor_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.Q8_0,
        ):
            raise ValueError(
                f"Cannot handle type {tensor.tensor_type.name} for tensor {repr(tensor.name)}"
            )
    print(f"* Preparing to convert from {file_endian.upper()} to {order.upper()}")
    print("\n*** Warning *** Warning *** Warning **")
    print("* This conversion process may damage the file. Ensure you have a backup.")
    print(
        "* The file will be modified immediately, so if conversion fails or is interrupted"
    )
    print("* the file will be corrupted. If you are positive then type YES:")
    response = input("YES, I am sure> ")
    if response != "YES":
        print("You didn't enter YES. Okay then, see ya!")
        sys.exit(0)
    print(f"\n* Converting fields ({len(reader.fields)})")
    for idx, field in enumerate(reader.fields.values()):
        print(
            f"- {idx:4}: Converting field {repr(field.name)}, part count: {len(field.parts)}"
        )
        for part in field.parts:
            part.byteswap(inplace=True)
    print(f"\n* Converting tensors ({len(reader.tensors)})")
    for idx, tensor in enumerate(reader.tensors):
        print(
            f"  - {idx:4}: Converting tensor {repr(tensor.name)}, type={tensor.tensor_type.name}, elements={tensor.n_elements}... ",
            end="",
        )
        tensor_type = tensor.tensor_type
        for part in tensor.field.parts:
            part.byteswap(inplace=True)
        if tensor_type != gguf.GGMLQuantizationType.Q8_0:
            tensor.data.byteswap(inplace=True)
            print()
            continue
        # A Q8_0 block consists of a f16 delta followed by 32 int8 quants, so 34 bytes
        block_size = 34
        n_blocks = len(tensor.data) // block_size
        for block_num in range(n_blocks):
            block_offs = block_num * block_size
            # I know I said f16, but it doesn't matter here - any simple 16 bit type works.
            delta = tensor.data[block_offs : block_offs + 2].view(dtype=np.uint16)
            delta.byteswap(inplace=True)
            if block_num % 100000 == 0:
                print(f"[{(n_blocks - block_num) // 1000}K]", end="")
                sys.stdout.flush()
        print()
    print("* Completion")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "convert_endian: Error: Missing arguments. Syntax: modify_gguf.py <filename> <little | big | native>",
            file=sys.stderr,
        )
        sys.exit(1)
    convert_byteorder(sys.argv[1], sys.argv[2])
