#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

# reuse model definitions from convert_hf_to_gguf.py
from convert_hf_to_gguf import Model

logger = logging.getLogger("lora-to-gguf")


def get_base_tensor_name(lora_tensor_name: str) -> str:
    base_name = lora_tensor_name.replace("base_model.model.", "")
    base_name = base_name.replace(".lora_A.weight", ".weight")
    base_name = base_name.replace(".lora_B.weight", ".weight")
    return base_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface PEFT LoRA adapter to a GGML compatible file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--base", type=Path, required=True,
        help="directory containing base model file",
    )
    parser.add_argument(
        "lora_path", type=Path,
        help="directory containing LoRA adapter file",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }
    ftype = ftype_map[args.outtype]

    dir_base_model = args.base
    dir_lora = args.lora_path
    input_json = os.path.join(dir_lora, "adapter_config.json")
    input_model = os.path.join(dir_lora, "adapter_model.bin")
    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_lora / 'ggml-lora-{ftype}.gguf'

    if os.path.exists(input_model):
        lora_model = torch.load(input_model, map_location="cpu")
    else:
        input_model = os.path.join(dir_lora, "adapter_model.safetensors")
        # lazy import load_file only if lora is in safetensors format.
        from safetensors.torch import load_file
        lora_model = load_file(input_model, device="cpu")

    # load base model
    logger.info(f"Loading base model: {dir_base_model.name}")
    hparams = Model.load_hparams(dir_base_model)
    with torch.inference_mode():
        try:
            model_class = Model.from_model_architecture(hparams["architectures"][0])
        except NotImplementedError:
            logger.error(f"Model {hparams['architectures'][0]} is not supported")
            sys.exit(1)

        model_instance = model_class(dir_base_model, ftype, fname_out, args.bigendian, False, False, None)
        logger.info("Set model parameters")
        model_instance.set_gguf_parameters()

        # adapter_config = json.load(input_json)
        model_instance.gguf_writer.add_string("training.type", "finetune_lora")
        if not model_instance.support_lora():
            logger.error("LoRA conversion is not yet supported for this model")
            sys.exit(1)

    # map original name to gguf name
    map_name: dict[str, str] = {}
    for tensor_name, tensor in lora_model.items():
        base_name = get_base_tensor_name(tensor_name)
        is_lora_a = ".lora_A.weight" in tensor_name
        is_lora_b = ".lora_B.weight" in tensor_name
        if not is_lora_a and not is_lora_b:
            logger.error(f"Unexpected name '{tensor_name}': Not a lora_A or lora_B tensor")
            sys.exit(1)
        dest_name = model_instance.map_tensor_name(base_name)
        dest_name = f"{dest_name}.lora_a" if is_lora_a else f"{dest_name}.lora_b"
        map_name[tensor_name] = dest_name

    # overwrite method
    def map_tensor_name(self, name: str) -> str:
        return map_name[name]

    # overwrite method
    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, tensor in lora_model.items():
            yield (name, tensor)

    # overwrite method
    def extra_f16_tensors(self, name: str, new_name: str, bid: int | None, n_dims: int) -> bool:
        del name, new_name, bid, n_dims  # unused
        return ftype != gguf.LlamaFileType.ALL_F32

    model_instance._map_tensor_name = model_instance.map_tensor_name # type: ignore
    model_instance.map_tensor_name = types.MethodType(map_tensor_name, model_instance)

    model_instance._get_tensors = model_instance.get_tensors # type: ignore
    model_instance.get_tensors = types.MethodType(get_tensors, model_instance)

    model_instance._extra_f16_tensors = model_instance.extra_f16_tensors # type: ignore
    model_instance.extra_f16_tensors = types.MethodType(extra_f16_tensors, model_instance)

    model_instance.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    logger.info("Exporting model...")
    model_instance.write()
    logger.info(f"Model successfully exported to {model_instance.fname_out}")
