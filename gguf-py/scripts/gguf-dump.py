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
    words = name.split(".")

    # Source: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-tensor-names
    abbreviation_dictionary = {
        'token_embd': 'Token embedding',
        'pos_embd': 'Position embedding',
        'output_norm': 'Output normalization',
        'output': 'Output',
        'attn_norm': 'Attention normalization',
        'attn_norm_2': 'Attention normalization',
        'attn_qkv': 'Attention query-key-value',
        'attn_q': 'Attention query',
        'attn_k': 'Attention key',
        'attn_v': 'Attention value',
        'attn_output': 'Attention output',
        'ffn_norm': 'Feed-forward network normalization',
        'ffn_up': 'Feed-forward network "up"',
        'ffn_gate': 'Feed-forward network "gate"',
        'ffn_down': 'Feed-forward network "down"',
        'ffn_gate_inp': 'Expert-routing layer for the Feed-forward network in Mixture of Expert models',
        'ffn_gate_exp': 'Feed-forward network "gate" layer per expert in Mixture of Expert models',
        'ffn_down_exp': 'Feed-forward network "down" layer per expert in Mixture of Expert models',
        'ffn_up_exp': 'Feed-forward network "up" layer per expert in Mixture of Expert models',
        'ssm_in': 'State space model input projections',
        'ssm_conv1d': 'State space model rolling/shift',
        'ssm_x': 'State space model selective parametrization',
        'ssm_a': 'State space model state compression',
        'ssm_d': 'State space model skip connection',
        'ssm_dt': 'State space model time step',
        'ssm_out': 'State space model output projection',
        'blk': 'Block'
    }

    expanded_words = []
    for word in words:
        word_norm = word.strip().lower()
        if word_norm in abbreviation_dictionary:
            expanded_words.append(abbreviation_dictionary[word_norm].title())
        else:
            expanded_words.append(word.title())

    return ' '.join(expanded_words)


def dump_markdown_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    markdown_content = ""
    markdown_content += f'# {args.model} - GGUF Internal File Dump\n'
    markdown_content += f'* Endian: {file_endian} endian\n'
    markdown_content += '\n'
    markdown_content += '## Key Value Metadata Store\n'
    markdown_content += f'There is {len(reader.fields)} key/value pair(s) in this file\n'
    markdown_content += '\n'

    markdown_content += '| POS | TYPE       | Elements | Key                                    | Value                                                                          |\n'
    markdown_content += '|-----|------------|----------|----------------------------------------|--------------------------------------------------------------------------------|\n'

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
        markdown_content += f'| {n:3} | {pretty_type:10} | {len(field.data):8} | {field.name:38} | {value:<78} |\n'

    markdown_content += "\n"

    if not args.no_tensors:
        # Group tensors by their prefix and maintain order
        tensor_prefix_order = []
        tensor_name_to_key = {}
        tensor_groups = {}
        total_elements = sum(tensor.n_elements for tensor in reader.tensors)

        for key, tensor in enumerate(reader.tensors):
            tensor_components = tensor.name.split('.')
            tensor_prefix = tensor_components[0]

            if tensor_prefix == 'blk':
                tensor_prefix = f"{tensor_components[0]}.{tensor_components[1]}"

            if tensor_prefix not in tensor_groups:
                tensor_groups[tensor_prefix] = []
                tensor_prefix_order.append(tensor_prefix)

            tensor_name_to_key[tensor.name] = key
            tensor_groups[tensor_prefix].append(tensor)

        # Tensors Mapping Dump
        markdown_content += f'## Tensors Overview {element_count_rounded_notation(total_elements)} Elements\n'
        markdown_content += f'Total number of elements in all tensors: {total_elements} Elements\n'
        markdown_content += '\n'

        for group in tensor_prefix_order:
            tensors = tensor_groups[group]
            group_elements = sum(tensor.n_elements for tensor in tensors)
            markdown_content += f"- [{translate_tensor_name(group)} Tensor Group - {element_count_rounded_notation(group_elements)} Elements](#{group.replace('.', '_')})\n"

        markdown_content += "\n"

        for group in tensor_prefix_order:
            tensors = tensor_groups[group]
            group_elements = sum(tensor.n_elements for tensor in tensors)
            group_percentage = group_elements / total_elements * 100
            markdown_content += f"### {translate_tensor_name(group)} Tensor Group : {element_count_rounded_notation(group_elements)} Elements <a name=\"{group.replace('.', '_')}\"></a>\n"
            markdown_content += "| T_ID | Tensor Layer Name         | Human Friendly Tensor Layer Name                   | Elements       | Shape                           | Type |\n"
            markdown_content += "|------|---------------------------|----------------------------------------------------|----------------|---------------------------------|------|\n"

            for tensor in tensors:
                human_friendly_name = translate_tensor_name(tensor.name.replace(".weight", ".(W)").replace(".bias", ".(B)"))
                prettydims = ' x '.join('{0:^5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
                markdown_content += f"| {tensor_name_to_key[tensor.name]:4} | {tensor.name:25} | {human_friendly_name:50} | ({element_count_rounded_notation(tensor.n_elements):>4}) {tensor.n_elements:7} | [{prettydims:29}] | {tensor.tensor_type.name:4} |\n"
            markdown_content += "\n"
            markdown_content += f"- Total elements in {group}: ({element_count_rounded_notation(group_elements):>4}) {group_elements}\n"
            markdown_content += f"- Percentage of total elements: {group_percentage:.2f}%\n"
            markdown_content += "\n\n"

        print(markdown_content)  # noqa: NP100


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
