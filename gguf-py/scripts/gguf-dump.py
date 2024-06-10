#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFValueType  # noqa: E402

logger = logging.getLogger("gguf-dump")


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)


# For more information about what field.parts and field.data represent,
# please see the comments in the modify_gguf.py example.
def dump_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    print(f'* File is {file_endian} endian, script is running on a {host_endian} endian host.')  # noqa: NP100
    print(f'* Dumping {len(reader.fields)} key/value pair(s)')  # noqa: NP100
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)

        log_message = f'  {n:5}: {pretty_type:10} | {len(field.data):8} | {field.name}'
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                log_message += ' = {0}'.format(repr(str(bytes(field.parts[-1]), encoding='utf-8')[:60]))
            elif field.types[0] in reader.gguf_scalar_to_np:
                log_message += ' = {0}'.format(field.parts[-1][0])
        print(log_message)  # noqa: NP100
    if args.no_tensors:
        return
    print(f'* Dumping {len(reader.tensors)} tensor(s)')  # noqa: NP100
    for n, tensor in enumerate(reader.tensors, 1):
        prettydims = ', '.join('{0:5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
        print(f'  {n:5}: {tensor.n_elements:10} | {prettydims} | {tensor.tensor_type.name:7} | {tensor.name}')  # noqa: NP100


def dump_metadata_json(reader: GGUFReader, args: argparse.Namespace) -> None:
    import json
    host_endian, file_endian = get_file_host_endian(reader)
    metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    result = {
        "filename": args.model,
        "endian": file_endian,
        "metadata": metadata,
        "tensors": tensors,
    }
    for idx, field in enumerate(reader.fields.values()):
        curr: dict[str, Any] = {
            "index": idx,
            "type": field.types[0].name if field.types else 'UNKNOWN',
            "offset": field.offset,
        }
        metadata[field.name] = curr
        if field.types[:1] == [GGUFValueType.ARRAY]:
            curr["array_types"] = [t.name for t in field.types][1:]
            if not args.json_array:
                continue
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                curr["value"] = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
            else:
                curr["value"] = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            curr["value"] = str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            curr["value"] = field.parts[-1].tolist()[0]
    if not args.no_tensors:
        for idx, tensor in enumerate(reader.tensors):
            tensors[tensor.name] = {
                "index": idx,
                "shape": tensor.shape.tolist(),
                "type": tensor.tensor_type.name,
                "offset": tensor.field.offset,
            }
    json.dump(result, sys.stdout)


def element_count_rounded_notation(count: int) -> str:
    if count > 1e15 :
        # Quadrillion
        scaled_amount = count * 1e-15
        scale_suffix = "Q"
    elif count > 1e12 :
        # Trillions
        scaled_amount = count * 1e-12
        scale_suffix = "T"
    elif count > 1e9 :
        # Billions
        scaled_amount = count * 1e-9
        scale_suffix = "B"
    elif count > 1e6 :
        # Millions
        scaled_amount = count * 1e-6
        scale_suffix = "M"
    elif count > 1e3 :
        # Thousands
        scaled_amount = count * 1e-3
        scale_suffix = "K"
    else:
        # Under Thousands
        scaled_amount = count
        scale_suffix = ""
    return f"{'~' if count > 1e3 else ''}{round(scaled_amount)}{scale_suffix}"


def translate_tensor_name(name):
    import re
    words = re.split(r"[._]", name)

    abbreviation_dictionary = {
        'ffn' : 'Feed Forward',
        'attn' : 'Attention',
        'blk' : 'Block',
        'norm' : 'Normalization',
        'embd' : 'Embedding',
    }

    expanded_words = []
    for word in words:
        word_norm = word.strip().lower()
        if word_norm in abbreviation_dictionary:
            expanded_words.append(abbreviation_dictionary[word_norm])
        else:
            expanded_words.append(word.title())

    return ' '.join(expanded_words)


def dump_markdown_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    print(f'# {args.model} - GGUF Internal File Dump')  # noqa: NP100
    print(f'* Endian: {file_endian} endian')  # noqa: NP100
    print('')  # noqa: NP100
    print('## Key Value Metadata Store')  # noqa: NP100
    print(f'There is {len(reader.fields)} key/value pair(s) in this file')  # noqa: NP100
    print('')  # noqa: NP100

    print('| POS | TYPE       | Elements | Key                                    | Value                                                                          |')  # noqa: NP100
    print('|-----|------------|----------|----------------------------------------|--------------------------------------------------------------------------------|')  # noqa: NP100

    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)

        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                value = repr(str(bytes(field.parts[-1]), encoding='utf-8')[:60])
            elif field.types[0] in reader.gguf_scalar_to_np:
                value = field.parts[-1][0]
        print(f'| {n:3} | {pretty_type:10} | {len(field.data):8} | {field.name:38} | {value:<78} |')  # noqa: NP100

    print("\n")  # noqa: NP100

    if not args.no_tensors:
        # Group tensors by their prefix and maintain order
        tensor_prefix_order = []
        tensor_groups = {}
        total_elements = sum(tensor.n_elements for tensor in reader.tensors)

        for tensor in reader.tensors:
            tensor_name = tensor.name.replace(".weight", "")
            tensor_components = tensor_name.split('.')
            tensor_prefix = tensor_components[0]
            if tensor_prefix == 'blk':
                tensor_prefix = f"{tensor_components[0]}.{tensor_components[1]}"

            if tensor_prefix not in tensor_groups:
                tensor_groups[tensor_prefix] = []
                tensor_prefix_order.append(tensor_prefix)

            tensor_groups[tensor_prefix].append(tensor)

        # Generate Markdown metadata
        for group in tensor_prefix_order:
            tensors = tensor_groups[group]
            group_elements = sum(tensor.n_elements for tensor in tensors)
            group_percentage = group_elements / total_elements * 100

            print(f"## {translate_tensor_name(group)} Tensor Group : {element_count_rounded_notation(group_elements)} Elements")  # noqa: NP100
            print("| Tensor Name          | Human Friendly Name                 |  Elements      | Shape                           | Type |")  # noqa: NP100
            print("|----------------------|-------------------------------------|----------------|---------------------------------|------|")  # noqa: NP100

            for tensor in tensors:
                tensor_name = tensor.name.replace(".weight", "")
                human_friendly_name = translate_tensor_name(tensor.name.replace(".weight", ""))
                prettydims = ' x '.join('{0:^5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
                print(f"| {tensor_name:20} | {human_friendly_name:35} | ({element_count_rounded_notation(tensor.n_elements):>4}) {tensor.n_elements:7} | [{prettydims:29}] | {tensor.tensor_type.name:4} |")  # noqa: NP100
            print("")  # noqa: NP100
            print(f"- Total elements in {group}: ({element_count_rounded_notation(group_elements):>4}) {group_elements}")  # noqa: NP100
            print(f"- Percentage of total elements: {group_percentage:.2f}%")  # noqa: NP100
            print("\n")  # noqa: NP100


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GGUF file metadata")
    parser.add_argument("model",           type=str,            help="GGUF format model filename")
    parser.add_argument("--no-tensors", action="store_true", help="Don't dump tensor metadata")
    parser.add_argument("--json",       action="store_true", help="Produce JSON output")
    parser.add_argument("--json-array", action="store_true", help="Include full array values in JSON output (long)")
    parser.add_argument("--markdown",   action="store_true", help="Produce markdown output")
    parser.add_argument("--verbose",    action="store_true", help="increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.json and not args.markdown:
        logger.info(f'* Loading: {args.model}')

    reader = GGUFReader(args.model, 'r')

    if args.json:
        dump_metadata_json(reader, args)
    elif args.markdown:
        dump_markdown_metadata(reader, args)
    else:
        dump_metadata(reader, args)


if __name__ == '__main__':
    main()
