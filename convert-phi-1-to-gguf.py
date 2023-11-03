#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py" / "gguf"))
import gguf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def check_required_files(directory: Path, required_files: List[str]) -> None:
    missing_files = [
        file_name
        for file_name in required_files
        if not (directory / file_name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")


def get_json_map(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r") as source_file:
        try:
            return json.load(source_file)
        except JSONDecodeError:
            raise ValueError(f"Failed to decode {file_path}")


def load_hyper_params(directory: Path, architecture: str) -> dict:
    config_path = directory / "config.json"
    hparams = get_json_map(config_path)

    # Ensure the expected architecture is present
    expected_architecture = architecture
    if hparams["architectures"][0] != expected_architecture:
        raise ValueError(
            f"Model architecture not supported: {hparams['architectures'][0]}"
        )

    return hparams


def initialize_writer(
    fname_out: str, architecture: str, ftype: str, hparams: Dict[str, Any]
) -> gguf.GGUFWriter:
    """
    Initializes the GGUF writer with the model metadata.

    :param fname_out: The filename for the output model.
    :param architecture: The model architecture enum name.
    :param ftype: The data type for the model file (e.g., 'F32', 'F16').
    :param hparams: The hyperparameters loaded from the model's config file.
    :return: An initialized GGUF writer object.
    """
    # Validate the architecture name
    if not hasattr(gguf.MODEL_ARCH, architecture):
        raise ValueError(f"Unsupported architecture: {architecture}")
    ARCH = getattr(gguf.MODEL_ARCH, architecture)

    # Validate the file type
    if ftype not in ['F32', 'F16']:
        raise ValueError(f"Unsupported file type: {ftype}")

    # Initialize the GGUF writer
    gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

    # Set the writer with the hyperparameters from MixFormerSequentialConfig
    gguf_writer.add_name(gguf.MODEL_ARCH_NAMES[ARCH])
    gguf_writer.add_context_length(hparams.get("n_positions", 2048))
    gguf_writer.add_embedding_length(hparams.get("n_embd", 1024))
    n_inner = hparams.get("n_inner", 4 * hparams.get("n_embd", 1024))
    gguf_writer.add_feed_forward_length(n_inner)
    gguf_writer.add_block_count(hparams.get("n_layer", 20))
    gguf_writer.add_head_count(hparams.get("n_head", 16))
    n_head_kv = hparams.get("n_head_kv", hparams.get("n_head", 16))
    gguf_writer.add_head_count_kv(n_head_kv)  # NOTE: arxiv:2203.11082
    gguf_writer.add_layer_norm_eps(hparams.get("layer_norm_epsilon", 1e-5))

    # Add the file type
    gguf_writer.add_file_type(ftype)

    return gguf_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Phi-1 model to a GGML compatible file"
    )
    parser.add_argument(
        "--vocab-only", action="store_true", help="extract only the vocab"
    )
    parser.add_argument(
        "--outfile", type=Path, help="path to write to; default: based on input"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "--ftype",
        type=str,
        choices=["f32", "f16"],
        default="f16",  # NOTE: Phi-1 is dtype float16.
        help="output format - use 'float32' for 32-bit tensors, 'float16' for 16-bit tensors",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()

        ftype = args.ftype
        directory = args.model  # Renamed for clarity

        if not directory.is_dir():
            raise NotADirectoryError(f"{directory} is not a directory.")

        required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        check_required_files(directory, required_files)

        # Reference the actual model file
        model = directory / "pytorch_model.bin"
        if not model.exists():
            raise FileNotFoundError(f"Model file {model} does not exist.")

        hparams = load_hyper_params(directory, "MixFormerSequentialForCausalLM")
        architecture = hparams["architectures"][0]

        if args.outfile is not None:
            fname_out = args.outfile
        else:
            fname_out = directory / f"ggml-model-{ftype}.gguf"

        if not fname_out.parent.exists():
            logging.warning(f"Output directory {fname_out.parent} does not exist.")

        gguf_writer = initialize_writer(fname_out, architecture, ftype, hparams)

        # Proceed with the model processing using the 'model' path
        # ... [rest of your existing code] ...

    except Exception as e:
        logging.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()


# # TOKENIZATION

# print("gguf: get tokenizer metadata")

# tokens: list[bytearray] = []
# scores: list[float] = []
# toktypes: list[int] = []

# # gpt2 tokenizer
# gguf_writer.add_tokenizer_model("gpt2")

# print("gguf: get gpt2 tokenizer vocab")

# # ref: https://github.com/cmp-nct/ggllm.cpp/blob/master/falcon_convert.py
# tokenizer = AutoTokenizer.from_pretrained(dir_model)

# # The number of tokens in tokenizer.json can differ from the expected vocab size.
# # This causes downstream issues with mismatched tensor sizes when running the inference
# vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
# assert max(tokenizer.vocab.values()) < vocab_size

# added_vocab = tokenizer.get_added_vocab()
# reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}

# for i in range(vocab_size):
#     if i not in reverse_vocab:
#         tokens.append(f"[PAD{i}]")
#         toktypes.append(gguf.TokenType.USER_DEFINED)
#     elif reverse_vocab[i] in added_vocab:
#         tokens.append(reverse_vocab[i])
#         if tokenizer.added_tokens_decoder[i].special:
#             toktypes.append(gguf.TokenType.CONTROL)
#         else:
#             toktypes.append(gguf.TokenType.USER_DEFINED)
#     else:
#         tokens.append(reverse_vocab[i])
#         toktypes.append(gguf.TokenType.NORMAL)

# gguf_writer.add_token_list(tokens)
# gguf_writer.add_token_types(toktypes)
# special_vocab = gguf.SpecialVocab(dir_model, load_merges=True, n_vocab=len(tokens))
# special_vocab.add_to_gguf(gguf_writer)

# # TENSORS

# tensor_map = gguf.get_tensor_name_map(ARCH, block_count)

# # params for qkv transform
# n_head = hparams["n_head"]
# n_head_kv = hparams["n_head_kv"] if "n_head_kv" in hparams else 1

# head_dim = hparams["n_embd"] // n_head

# # tensor info
# print("gguf: get tensor metadata")

# if num_parts == 0:
#     part_names = iter(("pytorch_model.bin",))
# else:
#     part_names = (
#         f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
#     )

# for part_name in part_names:
#     if args.vocab_only:
#         break
#     print("gguf: loading model part '" + part_name + "'")
#     model_part = torch.load(dir_model / part_name, map_location="cpu")

#     for name in model_part.keys():
#         data = model_part[name]

#         old_dtype = data.dtype

#         # convert any unsupported data types to float32
#         if data.dtype != torch.float16 and data.dtype != torch.float32:
#             data = data.to(torch.float32)

#         data = data.squeeze().numpy()

#         # map tensor names
#         new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
#         if new_name is None:
#             print("Can not map tensor '" + name + "'")
#             sys.exit()

#         n_dims = len(data.shape)
#         data_dtype = data.dtype

#         # if f32 desired, convert any float16 to float32
#         if ftype == 0 and data_dtype == np.float16:
#             data = data.astype(np.float32)

#         # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
#         if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
#             data = data.astype(np.float32)

#         # if f16 desired, convert any float32 2-dim weight tensors to float16
#         if (
#             ftype == 1
#             and data_dtype == np.float32
#             and name.endswith(".weight")
#             and n_dims == 2
#         ):
#             data = data.astype(np.float16)

#         print(
#             name,
#             "=>",
#             new_name
#             + ", shape = "
#             + str(data.shape)
#             + ", "
#             + str(old_dtype)
#             + " --> "
#             + str(data.dtype),
#         )

#         gguf_writer.add_tensor(new_name, data)


# print("gguf: write header")
# gguf_writer.write_header_to_file()
# print("gguf: write metadata")
# gguf_writer.write_kv_data_to_file()
# if not args.vocab_only:
#     print("gguf: write tensors")
#     gguf_writer.write_tensors_to_file()

# gguf_writer.close()

# print(f"gguf: model successfully exported to '{fname_out}'")
# print("")
